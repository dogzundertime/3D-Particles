--!native
--!strict

--[[
	- \\ this code owns the client-side blood pool and visuals
	- \\ instances are expensive and not thread safe, so all part creation, parenting, and cframe writes stay on the main client thread
	- \\ worker threads only touch raw numbers (sharedtable), so we avoid racey instance access and hidden engine sync costs

	- \\ actors do the stepping work so the main thread mostly just culls and writes cframes
	- \\ the expensive part is per-particle integration + collision decisions; actors let that run in parallel luau contexts
	- \\ renderstepped stays lean so camera, ui, and input don’t hitch even if splashes get spammed

	- \\ sharedtable is the handoff contract, so this script defines the schema and defends it from stale versions
	- \\ hot reloads / version drift can leave old sharedtables alive; one missing array can crash the render loop
	- \\ keeping schema here means workers can stay branch-free (no nil checks, no per-step validation)

	- \\ the goal is predictable frametime under worst case spam
	- \\ effects are “nice to have”; if the effect can tank fps it becomes a gameplay bug
	- \\ hard budgets (pool size, raycast caps, tick rates) keep worst-case cost bounded

	- \\ we intentionally drop particles instead of hitching the client
	- \\ saturation should fail quietly (fewer splats) instead of loudly (frame spikes, input lag)

	- \\ dataflow (how the pieces connect)
	- \\   1) Splash() reserves a free slot, writes all state arrays for that slot, then publishes it as active
	- \\   2) RenderStepped:
	- \\        - updateVisibility() runs at low hz and fills _visible so we can skip expensive work later
	- \\        - applyVisuals() runs every frame to reclaim dead slots and write cframes for visible ones
	- \\        - active slots are partitioned into visible / hidden / resting lists (cost follows usefulness)
	- \\   3) client collision helpers (single source of raycasts):
	- \\        - updateSupport() refreshes “floor support” so splats can lose support and fall off edges
	- \\        - updateWallsBudgeted() does budgeted sweeps and stores time-of-impact + normal for workers
	- \\   4) dispatchPhysics() sends dt + indices + small config to actor workers
	- \\   5) workers integrate and flip active=0 when lifetime ends
	- \\   6) applyVisuals() sees that flip and returns the slot to _free for constant-time reuse
]]

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local SharedTableRegistry = game:GetService("SharedTableRegistry")
local Workspace = game:GetService("Workspace")

export type Config = {
	PoolSize: number,
	Lifetime: number,

	SpawnMin: number,
	SpawnMax: number,

	WorkersFolderName: string,

	-- - \\ if nil we clone workers into the local playerscripts so it works in most setups
	-- - \\ actors must exist on this client to receive sendmessage, and cloning avoids manual place wiring
	WorkersCloneParent: Instance?,

	-- - \\ spend time where it actually shows
	-- - \\ visible particles get a higher stepping rate; hidden/resting get cheaper rates so we don’t waste work
	PhysicsHz: number,
	HiddenPhysicsHz: number,
	RestPhysicsHz: number,

	-- - \\ culling is separated so we don’t pay camera math every single frame
	-- - \\ visibility doesn’t change as fast as render frames, so sampling at ~10-20hz is usually enough
	CullHz: number,

	-- - \\ distance cap is a simple safety valve for giant maps
	MaxDrawDistance: number,

	-- - \\ stored here so workers don’t need to touch services
	GravityY: number,

	ArchUpMin: number,
	ArchUpMax: number,
	ArchOutMin: number,
	ArchOutMax: number,

	AirSpinDamp: number,
	GroundSpinDamp: number,

	SlideFriction: number,
	StopSpeed: number,

	SurfaceOffset: number,
	RaycastUp: number,
	RaycastDown: number,

	Template: BasePart?,
	FolderName: string,

	-- - \\ support refresh lets splats fall when their ledge support disappears
	SupportHz: number,
	SupportUp: number,
	SupportDown: number,
	SupportEps: number,

	-- - \\ sweeps prevent tunneling but are expensive, so they’re budgeted
	WallPad: number,
	WallMaxLen: number,

	-- - \\ abs of ny below this means “treat it like a side wall”
	WallNormalYMax: number,

	-- - \\ ny at or below negative this means “treat it like a ceiling” while airborne
	CeilingNormalYMin: number,

	-- - \\ small bounce keeps droplets from looking glued under ceilings and reads like splash energy
	CeilingBounce: number,

	-- - \\ particles are treated like spheres for cheap collision
	CollisionRadius: number,

	-- - \\ hard caps for raycasts per tick so worst case stays predictable
	MaxAirRaycasts: number,
	MaxSlideRaycasts: number,
}

type BloodSystem = {
	Config: Config,
	Initialized: boolean,

	-- - \\ sharedtable is the worker contract, kept flat and array-based
	-- - \\ struct-of-arrays avoids per-particle tables and keeps per-step loops tight and predictable
	_state: any,

	-- - \\ parts are owned only by the client thread; workers never touch instances
	_parts: { BasePart },

	-- - \\ cached visibility so we avoid writing cframes for offscreen items
	_visible: { [number]: boolean },

	-- - \\ dense active list keeps work proportional to what’s alive, not to pool size
	_activeList: { number },

	-- - \\ position map enables O(1) swap-remove during reclamation
	_activePos: { [number]: number },

	-- - \\ free stack gives constant-time allocation under spam
	_free: { number },

	_folder: Folder?,
	_actors: { Actor },
	_conn: RBXScriptConnection?,

	-- - \\ cached ray/overlap params so we don’t allocate every tick
	_map: Instance?,
	_rp: RaycastParams?,
	_op: OverlapParams?,
}

local ClientBloodSystem = {}
ClientBloodSystem.__index = ClientBloodSystem

-- - \\ key must match the worker key or the client and worker will talk past each other
-- - \\ versioning in the key is a cheap guard so old workers don’t interpret new schema
local STATE_KEY = "_BLOOD_STATE_V1"

-- - \\ minimal schema guard so stale sharedtables don’t crash the render loop after hot reload
-- - \\ we validate once at init so the hot loops can stay branch-free
local REQUIRED_FIELDS = {
	"active",
	"phase",
	"age",
	"px",
	"py",
	"pz",
	"vx",
	"vy",
	"vz",
	"lx",
	"ly",
	"lz",
	"nx",
	"ny",
	"nz",
	"ax",
	"ay",
	"az",
	"avx",
	"avy",
	"avz",
	"support",
	"wall",
	"wallt",
	"wallx",
	"wally",
	"wallz",
	"wallnx",
	"wallny",
	"wallnz",
}

local function nowCamera(): Camera?
	-- - \\ camera can be replaced by respawn or camera scripts, so always fetch current
	-- - \\ caching it once can go stale and lead to nil access or parenting into the wrong object
	return Workspace.CurrentCamera
end

local function ensureFolder(sys: BloodSystem): Folder
	-- - \\ keep all instances under one folder for easy cleanup and a tidy instance tree
	-- - \\ parenting under camera makes ownership obvious (client-only) and avoids cluttering workspace
	if sys._folder and sys._folder.Parent ~= nil then
		return sys._folder
	end

	local f = Instance.new("Folder")
	f.Name = sys.Config.FolderName

	local cam = nowCamera()
	if cam then
		f.Parent = cam
	else
		-- - \\ camera can be nil briefly during load, so don’t block init on it
		-- - \\ we’ll reparent later when the camera exists
		f.Parent = Workspace
	end

	sys._folder = f
	return f
end

local function makePartFromTemplate(sys: BloodSystem): BasePart
	-- - \\ pooling is the main lever for smoothness; clone/destroy churn causes gc hitches
	-- - \\ creating once and reusing means the “spam case” stays stable instead of spiking randomly
	local template = sys.Config.Template
	local p: BasePart

	if template then
		p = template:Clone()
	else
		local part = Instance.new("Part")
		part.Size = Vector3.new(1, 0.25, 1)
		part.Color = Color3.fromRGB(221, 45, 45)
		part.Material = Enum.Material.SmoothPlastic
		part.TopSurface = Enum.SurfaceType.Studs
		part.BottomSurface = Enum.SurfaceType.Inlet
		p = part
	end

	-- - \\ visuals only: shut off physics and queries to avoid hidden overhead and weird interactions
	-- - \\ this system “simulates” with numbers already, so letting physics/touch/query run would be double work
	p.Anchored = true
	p.CanCollide = false
	p.CanQuery = false
	p.CanTouch = false
	p.CastShadow = false

	-- - \\ start hidden to avoid one-frame flashes on creation/reuse
	-- - \\ when parts get parented or reused, a single bad frame is enough to look like a pop
	p.Transparency = 1
	p.Name = "Blood"

	return p
end

local function isSharedArray(t: any, poolSize: number): boolean
	-- - \\ accept plain tables for quick tests, but sharedtable is the real production path
	local tt = typeof(t)
	if tt ~= "SharedTable" and tt ~= "table" then
		return false
	end

	-- - \\ cheap “looks sized” check without scanning the whole array
	-- - \\ we only need a quick sanity check that indexing won’t explode later
	return t[poolSize] ~= nil
end

local function stateLooksValid(st: any, poolSize: number): boolean
	-- - \\ pool size mismatch breaks the index mapping immediately (part i <-> state arrays i)
	if typeof(st) ~= "SharedTable" and typeof(st) ~= "table" then
		return false
	end

	if type(st.poolSize) ~= "number" or st.poolSize ~= poolSize then
		return false
	end

	for _, k in ipairs(REQUIRED_FIELDS) do
		if not isSharedArray(st[k], poolSize) then
			return false
		end
	end

	return true
end

local function newArray(poolSize: number, initValue: number): any
	-- - \\ arrays keep per-particle state cheap and predictable for workers
	-- - \\ workers do tight numeric loops; tables-of-tables would add allocator churn and hash overhead
	local t = SharedTable.new()
	for i = 1, poolSize do
		t[i] = initValue
	end
	return t
end

local function buildState(poolSize: number): any
	-- - \\ single source of truth for the schema so client/worker don’t drift
	-- - \\ if schema drifts, you get “nothing moves” or “random nil” bugs that are painful to diagnose
	local st = SharedTable.new()

	st.poolSize = poolSize

	-- - \\ intentionally numbers only: no instances, no nested tables, no userdata
	st.active = newArray(poolSize, 0)
	st.phase = newArray(poolSize, 0)
	st.age = newArray(poolSize, 0)

	st.px = newArray(poolSize, 0)
	st.py = newArray(poolSize, 0)
	st.pz = newArray(poolSize, 0)

	st.vx = newArray(poolSize, 0)
	st.vy = newArray(poolSize, 0)
	st.vz = newArray(poolSize, 0)

	st.lx = newArray(poolSize, 0)
	st.ly = newArray(poolSize, 0)
	st.lz = newArray(poolSize, 0)

	st.nx = newArray(poolSize, 0)
	st.ny = newArray(poolSize, 1)
	st.nz = newArray(poolSize, 0)

	st.ax = newArray(poolSize, 0)
	st.ay = newArray(poolSize, 0)
	st.az = newArray(poolSize, 0)

	st.avx = newArray(poolSize, 0)
	st.avy = newArray(poolSize, 0)
	st.avz = newArray(poolSize, 0)

	st.support = newArray(poolSize, 0)

	st.wall = newArray(poolSize, 0)
	st.wallt = newArray(poolSize, 0)

	st.wallx = newArray(poolSize, 0)
	st.wally = newArray(poolSize, 0)
	st.wallz = newArray(poolSize, 0)

	st.wallnx = newArray(poolSize, 0)
	st.wallny = newArray(poolSize, 0)
	st.wallnz = newArray(poolSize, 0)

	return st
end

local function getOrCreateState(poolSize: number): any
	-- - \\ reuse a valid existing state so hot reloads don’t cause a hard visual reset mid-session
	-- - \\ this keeps “live edit” less jarring and avoids wasting time rebuilding if the schema matches
	local ok, existing = pcall(function()
		return SharedTableRegistry:GetSharedTable(STATE_KEY)
	end)

	if ok and existing and stateLooksValid(existing, poolSize) then
		return existing
	end

	local st = buildState(poolSize)

	-- - \\ only publish when the schema is complete and consistent
	-- - \\ registry is shared; publishing partial/invalid state means every worker reads garbage immediately
	SharedTableRegistry:SetSharedTable(STATE_KEY, st)
	return st
end

local function computeSurfaceBasis(nx: number, ny: number, nz: number): (Vector3, Vector3)
	-- - \\ grounded particles need a stable basis so they don’t yaw-flip on tiny normal noise
	-- - \\ degenerate normals can produce nan/inf if you normalize a near-zero vector
	local up = Vector3.new(nx, ny, nz)
	if up.Magnitude < 1e-6 then
		up = Vector3.new(0, 1, 0)
	else
		up = up.Unit
	end

	local ref = Vector3.new(0, 1, 0)
	local right = up:Cross(ref)
	if right.Magnitude < 1e-6 then
		-- - \\ if up is basically parallel to world up, pick another axis so the cross product isn’t ~0
		ref = Vector3.new(1, 0, 0)
		right = up:Cross(ref)
	end

	right = right.Unit
	return right, up
end

local function makeSurfaceCFrame(px: number, py: number, pz: number, nx: number, ny: number, nz: number): CFrame
	-- - \\ frommatrix is deterministic when you mostly know “up”
	-- - \\ lookat can twist unpredictably because forward is underconstrained
	local pos = Vector3.new(px, py, pz)
	local right, up = computeSurfaceBasis(nx, ny, nz)
	local back = right:Cross(up)
	return CFrame.fromMatrix(pos, right, up, back)
end

local function defaultConfig(): Config
	-- - \\ safe defaults: decent visuals, but bounded worst-case work
	-- - \\ the expensive bits are raycasts and cframe writes, so budgets are set around those
	return {
		PoolSize = 800,
		Lifetime = 8,

		SpawnMin = 1,
		SpawnMax = 3,

		WorkersFolderName = "BloodWorkers",
		WorkersCloneParent = nil,

		PhysicsHz = 60,
		HiddenPhysicsHz = 20,
		RestPhysicsHz = 10,
		CullHz = 15,

		MaxDrawDistance = 250,

		GravityY = -Workspace.Gravity,

		ArchUpMin = 18,
		ArchUpMax = 28,
		ArchOutMin = 6,
		ArchOutMax = 12,

		AirSpinDamp = 0.55,
		GroundSpinDamp = 7.5,

		SlideFriction = 3.5,
		StopSpeed = 0.02,

		SurfaceOffset = 0.02,
		RaycastUp = 1.5,
		RaycastDown = 12,

		Template = nil,
		FolderName = "_ClientBlood",

		-- - \\ support checks are frequent, so keep them lean
		SupportHz = 12,
		SupportUp = 0.35,
		SupportDown = 10,
		SupportEps = 0.12,

		WallPad = 0.20,
		WallMaxLen = 12.0,
		WallNormalYMax = 0.80,

		CeilingNormalYMin = 0.35,
		CeilingBounce = 0.35,

		CollisionRadius = 0.60,

		MaxAirRaycasts = 220,
		MaxSlideRaycasts = 120,
	}
end

function ClientBloodSystem.new(config: Config?): BloodSystem
	-- - \\ copy config once so callers can’t mutate tuning mid-sim and create weird desync behavior
	-- - \\ this also keeps profiling consistent because parameters don’t drift during runtime
	local cfg = defaultConfig()
	if config then
		for k, v in pairs(config :: any) do
			(cfg :: any)[k] = v
		end
	end

	local self: BloodSystem = setmetatable({
		Config = cfg,
		Initialized = false,

		_state = nil,
		_parts = {},
		_visible = {},
		_activeList = {},
		_activePos = {},
		_free = {},

		_folder = nil,
		_actors = {},
		_conn = nil,

		_map = nil,
		_rp = nil,
		_op = nil,
	}, ClientBloodSystem)

	return self
end

function ClientBloodSystem:_cloneWorkers(): { Actor }
	-- - \\ cloning locally keeps deployment simple and guarantees the actors exist in this client environment
	-- - \\ it also avoids ordering problems where the effect runs before some replicated setup finishes
	local workersFolder = ReplicatedStorage:WaitForChild(self.Config.WorkersFolderName) :: Folder

	local player = Players.LocalPlayer
	local parent = self.Config.WorkersCloneParent
	if parent == nil then
		parent = player:WaitForChild("PlayerScripts")
	end

	local cloned = workersFolder:Clone()
	cloned.Name = "_BloodWorkers_Cloned"
	cloned.Parent = parent

	local actors = {}
	for _, child in ipairs(cloned:GetChildren()) do
		if child:IsA("Actor") then
			table.insert(actors, child)
		end
	end

	return actors
end

local function popFree(sys: BloodSystem): number?
	-- - \\ stack pop keeps allocation constant-time under spam
	-- - \\ the “empty” case is intentional: it tells the caller to drop particles instead of doing expensive work
	local n = #sys._free
	if n == 0 then
		return nil
	end

	local idx = sys._free[n]
	sys._free[n] = nil
	return idx
end

local function resolveSpawn(sys: BloodSystem, start: Vector3): Vector3
	-- - \\ spawning inside geometry creates extra raycasts and ugly pop-out artifacts
	-- - \\ a small overlap-based nudge is bounded cost and usually “good enough” for a visual effect
	local op = sys._op
	if not op then
		return start
	end

	local r = math.max(0.15, sys.Config.CollisionRadius * 0.90)

	local function occupied(p: Vector3): boolean
		-- - \\ MaxParts=1 gives a cheap “any hit” test instead of building a big list
		local parts = Workspace:GetPartBoundsInRadius(p, r, op)
		return parts[1] ~= nil
	end

	if not occupied(start) then
		return start
	end

	-- - \\ bounded search: a few offsets, a few downward steps, then give up
	-- - \\ the goal isn’t perfect placement, it’s “avoid the worst case” without spiking cost
	local downStep = 0.25
	local maxDown = 6.0
	local lateral = 0.35

	local offsets = {
		Vector3.new(0, 0, 0),
		Vector3.new(lateral, 0, 0),
		Vector3.new(-lateral, 0, 0),
		Vector3.new(0, 0, lateral),
		Vector3.new(0, 0, -lateral),
		Vector3.new(lateral, 0, lateral),
		Vector3.new(-lateral, 0, lateral),
		Vector3.new(lateral, 0, -lateral),
		Vector3.new(-lateral, 0, -lateral),
	}

	local steps = math.floor(maxDown / downStep)
	for i = 1, steps do
		local base = start - Vector3.new(0, downStep * i, 0)
		for j = 1, #offsets do
			local cand = base + offsets[j]
			if not occupied(cand) then
				return cand
			end
		end
	end

	-- - \\ downward bias usually reads more natural than shoving sideways
	return start - Vector3.new(0, sys.Config.CollisionRadius, 0)
end

function ClientBloodSystem:Init()
	-- - \\ lazy init so we only pay pool allocation if the effect is actually used
	-- - \\ this keeps initial load/menus smoother and avoids wasting memory if nothing ever splashes
	if self.Initialized then
		return
	end

	assert(RunService:IsClient(), "ClientBloodSystem must run on the client")

	-- - \\ build/reuse state here so workers can assume arrays exist and stay branch-free
	self._state = getOrCreateState(self.Config.PoolSize)

	-- - \\ cache map + params so the hot path is mostly math, not allocations
	-- - \\ raycast/overlap params are objects; rebuilding them inside loops creates avoidable churn and gc pressure
	local mapFolder = Workspace:WaitForChild("MAP")
	self._map = mapFolder

	local rp = RaycastParams.new()
	rp.FilterType = Enum.RaycastFilterType.Include
	rp.FilterDescendantsInstances = { mapFolder }
	rp.IgnoreWater = true
	self._rp = rp

	local op = OverlapParams.new()
	op.FilterType = Enum.RaycastFilterType.Include
	op.FilterDescendantsInstances = { mapFolder }
	op.MaxParts = 1
	op.RespectCanCollide = true
	self._op = op

	-- - \\ allocate parts once then reuse by toggling transparency and cframe
	-- - \\ this is the heart of the “no hitch under spam” design
	local folder = ensureFolder(self)
	table.clear(self._parts)
	table.clear(self._free)

	for i = 1, self.Config.PoolSize do
		local p = makePartFromTemplate(self)
		p.Parent = folder

		-- - \\ park far away while hidden so reuse never flashes at origin
		p.CFrame = CFrame.new(0, -1e6, 0)

		self._parts[i] = p
		self._visible[i] = false
		self._free[i] = i

		-- - \\ prefill arrays so workers never see nil if something interrupts init
		self._state.active[i] = 0
		self._state.phase[i] = 0
		self._state.age[i] = 0
		self._state.support[i] = 0
		self._state.wall[i] = 0
		self._state.wallt[i] = 0
	end

	-- - \\ actors do the heavy stepping; main thread stays focused on rendering and bookkeeping
	self._actors = self:_cloneWorkers()
	self.Initialized = true

	local physAccumVis = 0.0
	local physStepVis = 1 / math.max(1, self.Config.PhysicsHz)

	local physAccumHid = 0.0
	local physStepHid = 1 / math.max(1, self.Config.HiddenPhysicsHz)

	local physAccumRest = 0.0
	local physStepRest = 1 / math.max(1, self.Config.RestPhysicsHz)

	local cullAccum = 0.0
	local cullStep = 1 / math.max(1, self.Config.CullHz)

	local camDistSq = self.Config.MaxDrawDistance * self.Config.MaxDrawDistance

	local supportAccum = 0.0
	local supportStep = 1 / math.max(1, self.Config.SupportHz)

	local function updateVisibility()
		-- - \\ visibility is the gate that decides if we spend cframe cost
		-- - \\ cframe writes are expensive, so this function exists mostly to help other code skip work
		local cam = nowCamera()
		if not cam then
			return
		end

		local vpSize = cam.ViewportSize
		local cx, cy, cz = cam.CFrame.X, cam.CFrame.Y, cam.CFrame.Z

		for _, idx in ipairs(self._activeList) do
			local px = self._state.px[idx]
			local py = self._state.py[idx]
			local pz = self._state.pz[idx]

			local dx = px - cx
			local dy = py - cy
			local dz = pz - cz
			local d2 = dx * dx + dy * dy + dz * dz

			local vis = false
			if d2 <= camDistSq then
				local v, onScreen = cam:WorldToViewportPoint(Vector3.new(px, py, pz))
				if onScreen and v.Z > 0 then
					-- - \\ margin reduces edge popping during small camera jitters/shake
					-- - \\ it’s cheap hysteresis without keeping extra per-particle state
					if v.X >= -50 and v.Y >= -50 and v.X <= (vpSize.X + 50) and v.Y <= (vpSize.Y + 50) then
						vis = true
					end
				end
			end

			if self._visible[idx] ~= vis then
				self._visible[idx] = vis

				-- - \\ transparency flips here so hidden parts stop consuming render cost
				-- - \\ the simulation continues, but the expensive render-side updates can be gated elsewhere
				self._parts[idx].Transparency = if vis then 0 else 1
			end
		end
	end

	local function applyVisuals()
		-- - \\ this runs every frame for two reasons:
		-- - \\   1) reclamation stays responsive, which refills _free quickly during spam
		-- - \\   2) visible cframe writes feel tight because they’re synced to RenderStepped
		-- - \\ workers only flip active=0; the client owns the instance pool and the bookkeeping
		for i = #self._activeList, 1, -1 do
			local idx = self._activeList[i]

			if self._state.active[idx] == 0 then
				local pos = self._activePos[idx]
				if pos then
					-- - \\ swap-remove avoids shifting the array when lots of particles die together
					-- - \\ shifting is the kind of “rare spike” that looks fine in normal tests but explodes under spam
					-- - \\ swap-remove keeps the cost basically constant no matter how many die at once
					local lastIdx = self._activeList[#self._activeList]
					self._activeList[pos] = lastIdx
					self._activePos[lastIdx] = pos
					self._activeList[#self._activeList] = nil
					self._activePos[idx] = nil
				end

				-- - \\ clear contact flags so reused slots don’t inherit phantom collisions
				-- - \\ reuse is common under spam; stale flags are a classic “random bug” source
				self._state.wall[idx] = 0
				self._state.wallt[idx] = 0
				self._state.support[idx] = 0

				local part = self._parts[idx]
				part.Transparency = 1
				part.CFrame = CFrame.new(0, -1e6, 0)
				self._visible[idx] = false

				-- - \\ return idx to the free stack so future spawns stay constant-time
				self._free[#self._free + 1] = idx
			else
				-- - \\ skip cframe writes when hidden; worker still simulates so state stays correct
				-- - \\ this is where _visible does real work: it prevents the hottest cost from scaling with “active”
				if self._visible[idx] then
					local px = self._state.px[idx]
					local py = self._state.py[idx]
					local pz = self._state.pz[idx]

					local ay = self._state.ay[idx]
					local phase = self._state.phase[idx]

					local cf: CFrame
					if phase == 1 then
						-- - \\ airborne gets free rotation so it reads like spray instead of a flat sticker
						local ax = self._state.ax[idx]
						local az = self._state.az[idx]
						cf = CFrame.new(px, py, pz) * CFrame.Angles(ax, ay, az)
					else
						-- - \\ grounded aligns to surface normal so it sits cleanly on slopes
						-- - \\ this also keeps grounded motion visually stable when sliding/settling
						local nx = self._state.nx[idx]
						local ny = self._state.ny[idx]
						local nz = self._state.nz[idx]
						cf = makeSurfaceCFrame(px, py, pz, nx, ny, nz) * CFrame.Angles(0, ay, 0)
					end

					self._parts[idx].CFrame = cf
				end
			end
		end
	end

	local function dispatchPhysics(dt: number, list: { number })
		-- - \\ split indices evenly across actors so one worker doesn’t become the bottleneck
		-- - \\ cost per particle varies (collisions, support loss, ceiling hits), so simple balancing works well
		local actors = self._actors
		local n = #actors
		if n == 0 then
			return
		end

		local activeCount = #list
		if activeCount == 0 then
			return
		end

		local per = math.max(1, math.floor((activeCount + n - 1) / n))

		for w = 1, n do
			local a = actors[w]
			local startPos = (w - 1) * per + 1
			if startPos > activeCount then
				break
			end
			local endPos = math.min(activeCount, startPos + per - 1)

			-- - \\ allocate one slice per worker per dispatch (bounded by worker count, not by particle count)
			-- - \\ this keeps messaging overhead stable even if activeCount is large
			local slice = table.create(endPos - startPos + 1)
			local s = 1
			for j = startPos, endPos do
				slice[s] = list[j]
				s += 1
			end

			-- - \\ message carries config so workers don’t depend on shared mutable tuning
			-- - \\ that keeps “state” focused on particle data only, and avoids weird bugs from config drifting mid-step
			a:SendMessage("Step", {
				dt = dt,
				indices = slice,

				gravityY = self.Config.GravityY,
				lifetime = self.Config.Lifetime,

				surfaceOffset = self.Config.SurfaceOffset,
				slideFriction = self.Config.SlideFriction,
				stopSpeed = self.Config.StopSpeed,

				airSpinDamp = self.Config.AirSpinDamp,
				groundSpinDamp = self.Config.GroundSpinDamp,

				ceilingNormalYMin = self.Config.CeilingNormalYMin,
				ceilingBounce = self.Config.CeilingBounce,
			})
		end
	end

	local function updateSupport(list: { number })
		-- - \\ support refresh prevents “hovering decals” when particles drift off ledges
		-- - \\ doing it on the client keeps raycast budgets single-sourced instead of multiplying by worker count
		local rpLocal = self._rp :: RaycastParams

		for i = 1, #list do
			local idx = list[i]
			local phase = self._state.phase[idx]

			-- - \\ inactive/resting don’t benefit from support checks, so skip them
			-- - \\ this keeps raycast spend focused on particles that might actually transition (air -> slide -> rest)
			if phase ~= 0 and phase ~= 3 then
				local px = self._state.px[idx]
				local py = self._state.py[idx]
				local pz = self._state.pz[idx]

				local origin = Vector3.new(px, py, pz) + Vector3.new(0, self.Config.SupportUp, 0)
				local dir = Vector3.new(0, -(self.Config.SupportUp + self.Config.SupportDown), 0)
				local hit = Workspace:Raycast(origin, dir, rpLocal)

				if hit then
					local hx, hy, hz = hit.Position.X, hit.Position.Y, hit.Position.Z
					local nx, ny, nz = hit.Normal.X, hit.Normal.Y, hit.Normal.Z

					-- - \\ worker wants the latest plane info without doing its own raycasts
					-- - \\ this keeps “what surface am i on” consistent across the pipeline
					self._state.lx[idx] = hx
					self._state.ly[idx] = hy
					self._state.lz[idx] = hz
					self._state.nx[idx] = nx
					self._state.ny[idx] = ny
					self._state.nz[idx] = nz

					local dx = px - hx
					local dy = py - hy
					local dz = pz - hz
					local sep = dx * nx + dy * ny + dz * nz

					-- - \\ project onto the normal for a stable “distance above plane” metric
					-- - \\ eps adds hysteresis so tiny noise doesn’t toggle support every tick
					self._state.support[idx] = if sep <= (self.Config.SurfaceOffset + self.Config.SupportEps) then 1 else 0
				else
					self._state.support[idx] = 0
				end
			end
		end
	end

	local function updateWallsBudgeted(list: { number }, dt: number)
		-- - \\ sweeps prevent tunneling; budgets keep worst case predictable
		-- - \\ airborne gets priority because it’s faster and tunneling looks worse there
		local rpLocal = self._rp :: RaycastParams

		local radius = self.Config.CollisionRadius
		local pad = self.Config.WallPad
		local maxLen = self.Config.WallMaxLen
		local nyMax = self.Config.WallNormalYMax
		local ceilMin = self.Config.CeilingNormalYMin

		local function tryOne(idx: number, phase: number): boolean
			-- - \\ don’t do multiple sweeps for the same particle in the same frame
			-- - \\ once a contact is stored, the worker resolves using that, so extra rays here are just wasted budget
			if self._state.wall[idx] ~= 0 then
				return false
			end

			local vx = self._state.vx[idx]
			local vy = self._state.vy[idx]
			local vz = self._state.vz[idx]

			local sp2 = vx * vx + vy * vy + vz * vz
			if sp2 <= 0.0025 then
				-- - \\ slow movers rarely tunnel, so save the raycast for cases where it matters
				return false
			end

			local sp = math.sqrt(sp2)
			local ux = vx / sp
			local uy = vy / sp
			local uz = vz / sp

			local len = sp * dt + radius + pad
			if len > maxLen then
				-- - \\ hard cap prevents huge rays during dt spikes or extreme speeds
				-- - \\ huge rays also tend to “see” geometry you didn’t mean to interact with (across rooms, behind walls)
				len = maxLen
			end

			local px = self._state.px[idx]
			local py = self._state.py[idx]
			local pz = self._state.pz[idx]

			local origin = Vector3.new(px, py, pz)
			local dir = Vector3.new(ux * len, uy * len, uz * len)

			local hit = Workspace:Raycast(origin, dir, rpLocal)
			if not hit then
				return false
			end

			local nY = hit.Normal.Y
			local isSide = math.abs(nY) < nyMax
			local isCeil = (phase == 1) and (nY <= -ceilMin)

			-- - \\ floors are handled by support logic; here we only care about side walls and ceilings
			-- - \\ mixing floor resolution into this path tends to create double-resolve jitter (support says “on floor”, wall says “hit floor”)
			if not (isSide or isCeil) then
				return false
			end

			local d = hit.Distance
			local stopDist = d - radius
			if stopDist < 0 then
				stopDist = 0
			end

			local t = stopDist / len
			if t < 0 then
				t = 0
			end
			if t > 1 then
				t = 1
			end

			-- - \\ store time-of-impact + contact so workers can resolve without doing their own raycasts
			-- - \\ this keeps raycast cost single-sourced and budgeted in one place
			self._state.wall[idx] = 1
			self._state.wallt[idx] = t

			self._state.wallx[idx] = px + ux * stopDist
			self._state.wally[idx] = py + uy * stopDist
			self._state.wallz[idx] = pz + uz * stopDist

			self._state.wallnx[idx] = hit.Normal.X
			self._state.wallny[idx] = hit.Normal.Y
			self._state.wallnz[idx] = hit.Normal.Z

			return true
		end

		local airBudget = self.Config.MaxAirRaycasts
		for i = 1, #list do
			if airBudget <= 0 then
				break
			end
			local idx = list[i]
			if self._state.phase[idx] == 1 then
				if tryOne(idx, 1) then
					airBudget -= 1
				end
			end
		end

		local slideBudget = self.Config.MaxSlideRaycasts
		for i = 1, #list do
			if slideBudget <= 0 then
				break
			end
			local idx = list[i]
			if self._state.phase[idx] == 2 then
				if tryOne(idx, 2) then
					slideBudget -= 1
				end
			end
		end
	end

	self._conn = RunService.RenderStepped:Connect(function(dt)
		-- - \\ order here matters for both feel and cost:
		-- - \\   - reparent folder first so visuals stay under the correct camera after swaps
		-- - \\   - cull before visuals so we don’t write cframes for things we’re about to hide
		-- - \\   - reclaim every frame so _free refills quickly and spam stays smooth
		-- - \\   - run support/walls before dispatch so workers get the freshest collision hints
		local cam = nowCamera()
		if cam and self._folder and self._folder.Parent ~= cam then
			-- - \\ camera swaps happen; reparent is cheap and avoids rebuilding the pool
			self._folder.Parent = cam
		end

		cullAccum += dt
		if cullAccum >= cullStep then
			cullAccum = 0
			updateVisibility()
		end

		applyVisuals()

		physAccumVis += dt
		physAccumHid += dt
		physAccumRest += dt

		local doVis = physAccumVis >= physStepVis
		local doHid = physAccumHid >= physStepHid
		local doRest = physAccumRest >= physStepRest

		if doVis or doHid or doRest then
			-- - \\ partitioning keeps expensive work focused on visible particles
			-- - \\ the point is not “perfect simulation everywhere”, it’s “spend where the player benefits”
			local visList = table.create(64)
			local hidList = table.create(64)
			local restList = table.create(64)

			local vN, hN, rN = 0, 0, 0

			for _, idx in ipairs(self._activeList) do
				local phase = self._state.phase[idx]
				if phase == 3 then
					rN += 1
					restList[rN] = idx
				elseif self._visible[idx] then
					vN += 1
					visList[vN] = idx
				else
					hN += 1
					hidList[hN] = idx
				end
			end

			if doVis then
				-- - \\ clamp dt so one hitch doesn’t turn into a huge “catch up” step
				-- - \\ big dt makes particles teleport, makes sweeps long, and tends to create weird one-frame contacts
				-- - \\ clamping keeps the sim stable and keeps sweep rays bounded even if the game stutters
				local stepDt = math.min(physAccumVis, 0.1)
				physAccumVis = 0

				supportAccum += stepDt
				if supportAccum >= supportStep then
					supportAccum = 0
					updateSupport(visList)
				end

				updateWallsBudgeted(visList, stepDt)
				dispatchPhysics(stepDt, visList)
			end

			if doHid then
				-- - \\ hidden still steps so it doesn’t freeze and then pop back wrong when visible again
				-- - \\ lower hz is enough to keep “where it should be” correct without paying full rate
				local stepDt = math.min(physAccumHid, 0.1)
				physAccumHid = 0
				dispatchPhysics(stepDt, hidList)
			end

			if doRest then
				-- - \\ resting is low motion; low rate saves a lot of work with almost no visual loss
				-- - \\ this is a big win because resting can become the majority of active particles over time
				local stepDt = math.min(physAccumRest, 0.1)
				physAccumRest = 0
				dispatchPhysics(stepDt, restList)
			end
		end
	end)
end

function ClientBloodSystem:Splash(at: Vector3 | BasePart)
	-- - \\ public api: make it safe even if the caller doesn’t know init order
	if not self.Initialized then
		self:Init()
	end

	local pos: Vector3
	if typeof(at) == "Instance" then
		pos = (at :: BasePart).Position
	else
		pos = at :: Vector3
	end

	local spawnCount = math.random(self.Config.SpawnMin, self.Config.SpawnMax)
	local folder = ensureFolder(self)
	local rpLocal = self._rp :: RaycastParams

	for _ = 1, spawnCount do
		-- - \\ if we’re saturated, drop particles instead of stalling the client
		-- - \\ resizing/scanning here would be the classic “spam makes fps die” bug
		local idx = popFree(self)
		if not idx then
			break
		end

		-- - \\ small jitter keeps repeated splashes from stacking perfectly and helps the occupancy nudge
		-- - \\ identical positions tend to create identical ray hits, which makes the effect look copy-pasted
		local jitter = Vector3.new(
			(math.random() - 0.5) * 0.25,
			(math.random() - 0.5) * 0.10,
			(math.random() - 0.5) * 0.25
		)

		-- - \\ keep starts out of geometry so collision recovery doesn’t waste work
		local start = resolveSpawn(self, pos + jitter)

		-- - \\ seed an initial support plane so the first rendered frames don’t pop orientation later
		-- - \\ waiting for the next support tick can leave you rendering with a default normal for a moment
		local upClamp = math.min(self.Config.RaycastUp, math.max(0.25, self.Config.CollisionRadius * 0.55))
		local origin = start + Vector3.new(0, upClamp, 0)
		local dir = Vector3.new(0, -(upClamp + self.Config.RaycastDown), 0)

		local hit = Workspace:Raycast(origin, dir, rpLocal)

		local lx, ly, lz = start.X, start.Y, start.Z
		local nx, ny, nz = 0, 1, 0
		if hit then
			lx, ly, lz = hit.Position.X, hit.Position.Y, hit.Position.Z
			nx, ny, nz = hit.Normal.X, hit.Normal.Y, hit.Normal.Z
		end

		-- - \\ random spray gives organic variation without authoring curves per weapon
		-- - \\ the min/max ranges keep it bounded so droplets don’t fly across the entire map
		local theta = math.random() * math.pi * 2
		local out = self.Config.ArchOutMin + (self.Config.ArchOutMax - self.Config.ArchOutMin) * math.random()
		local up = self.Config.ArchUpMin + (self.Config.ArchUpMax - self.Config.ArchUpMin) * math.random()

		local vx = math.cos(theta) * out
		local vz = math.sin(theta) * out
		local vy = up

		-- - \\ if we still overlap after resolve, bias downward so it reads like a splat instead of a sideways teleport
		-- - \\ sideways “escape” tends to look like it popped through a wall; downward reads like “hit something and fell”
		do
			local op = self._op
			if op then
				local parts = Workspace:GetPartBoundsInRadius(
					start,
					math.max(0.15, self.Config.CollisionRadius * 0.90),
					op
				)
				if parts[1] ~= nil then
					vy = -math.abs(vy) * 0.25
					vx *= 0.5
					vz *= 0.5
				end
			end
		end

		-- - \\ spin sells the spray, damping (in the worker) settles it after landing
		local avx = (math.random() - 0.5) * 30
		local avy = (math.random() - 0.5) * 44
		local avz = (math.random() - 0.5) * 30

		-- - \\ write a fully initialized slot before we treat it as active
		-- - \\ this is the “publish” moment: once active=1, the slot is fair game for every other system
		-- - \\     - the render loop may read px/py/pz immediately for visibility and cframe
		-- - \\     - support / wall sweeps can run this tick and expect position + velocity to be coherent
		-- - \\     - a worker can pick the index up on the next dispatch and integrate right away
		-- - \\ if we flip active first, anything that runs before the rest of the fields are filled
		-- - \\ can see half-written state (old values from a recycled slot or zeros) and do the wrong thing
		-- - \\ a few concrete failures this avoids:
		-- - \\     - one-frame teleport to an old location because px/pz still held recycled data
		-- - \\     - a sweep ray starting from the origin because position wasn’t written yet, producing random wall hits
		-- - \\     - the worker integrating with velocity = 0 for one step, making the droplet hang then snap
		-- - \\ treating active as a commit flag keeps the rest of the pipeline simple and fast
		self._state.phase[idx] = 1
		self._state.age[idx] = 0

		self._state.px[idx] = start.X
		self._state.py[idx] = start.Y
		self._state.pz[idx] = start.Z

		self._state.vx[idx] = vx
		self._state.vy[idx] = vy
		self._state.vz[idx] = vz

		self._state.lx[idx] = lx
		self._state.ly[idx] = ly
		self._state.lz[idx] = lz

		self._state.nx[idx] = nx
		self._state.ny[idx] = ny
		self._state.nz[idx] = nz

		-- - \\ random start orientation avoids obvious tiling/pattern repetition
		-- - \\ this is basically free variety: a few numbers that make repeated splashes feel less copy-paste
		self._state.ax[idx] = math.random() * math.pi * 2
		self._state.ay[idx] = math.random() * math.pi * 2
		self._state.az[idx] = math.random() * math.pi * 2

		self._state.avx[idx] = avx
		self._state.avy[idx] = avy
		self._state.avz[idx] = avz

		-- - \\ clear contact flags so the first worker step starts clean
		-- - \\ reuse is common; stale wall/support flags are the kind of bug that only shows up under heavy spam
		self._state.support[idx] = 0
		self._state.wall[idx] = 0
		self._state.wallt[idx] = 0

		-- - \\ publish active last on purpose
		-- - \\ this makes “active=1” mean “slot is complete and safe to read” for every consumer
		self._state.active[idx] = 1

		-- - \\ track it locally so the client can iterate only active slots and reclaim with swap-remove
		-- - \\ the pool is capacity; the active list is the real workload
		self._activeList[#self._activeList + 1] = idx
		self._activePos[idx] = #self._activeList

		-- - \\ part stays hidden until visibility sampling marks it visible
		-- - \\ that keeps “spawn spam” from instantly turning into “cframe spam”
		self._parts[idx].Parent = folder
		self._parts[idx].Transparency = 1
		self._visible[idx] = false
	end
end

function ClientBloodSystem:Destroy()
	-- - \\ explicit teardown avoids leaked connections and pooled instances during mode switches or reloads
	-- - \\ renderstepped connections keep running if you forget them, and those leaks become “mystery fps drain”
	if self._conn then
		self._conn:Disconnect()
		self._conn = nil
	end

	for _, p in ipairs(self._parts) do
		if p then
			p:Destroy()
		end
	end

	self._parts = {}
	self._activeList = {}
	self._activePos = {}
	self._free = {}
	self._actors = {}

	if self._folder then
		self._folder:Destroy()
		self._folder = nil
	end

	self.Initialized = false
end

return ClientBloodSystem
