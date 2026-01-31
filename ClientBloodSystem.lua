--!native
--!strict

--[[
	- \\ this file is the client side owner of the blood pool and visuals
	- \\ why: instances are expensive + not thread safe, so we keep all part creation, parenting, and cframe writes on the main client thread
	- \\ why: the worker threads should only touch raw numbers (sharedtable), otherwise you risk racey instance access + hidden engine sync costs

	- \\ actors do the stepping work so the main thread mostly just culls and writes cframes
	- \\ why: the expensive part is the per-particle physics-ish integration and collision decisions; actors let that run in parallel luau contexts
	- \\ why: renderstepped should be kept lean so camera + ui + input never hitch, even if you spam splashes

	- \\ sharedtable is the handoff contract so this script builds the schema and defends it from stale versions
	- \\ why: hot reloads / place version drift can leave old sharedtables alive; if we assume fields exist, one missing array can crash the render loop
	- \\ why: schema lives here so workers can be dumb-fast (no branching for nil / no per-step validation)

	- \\ the main goal is predictable frametime even if splash gets spammed
	- \\ why: effects are "nice to have"; if the effect can tank fps then it becomes a gameplay bug
	- \\ why: the design uses hard budgets (pool size, raycast caps, tick rates) so worst-case cost is bounded

	- \\ we would rather drop particles than hitch the client
	- \\ why: saturation behavior should fail "quietly" (fewer splats) instead of failing "loudly" (frame spikes, input lag)
	- \\ why: a stable 60/120 fps with fewer particles looks better than 5 fps with perfect particles
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
	-- - \\ why: actors must exist on the local client to receive sendmessage; cloning avoids needing manual place wiring
	-- - \\ why: playerscripts survives respawns and is a safe client-only container
	WorkersCloneParent: Instance?,

	-- - \\ these rates let us spend the most time on what the player can actually see
	-- - \\ why: visible particles get higher stepping rate to look smooth; hidden/resting get cheaper rates to avoid wasted work
	-- - \\ why: variable rate is the easiest multiplier on performance without changing visual density
	PhysicsHz: number,
	HiddenPhysicsHz: number,
	RestPhysicsHz: number,

	-- - \\ culling is separated so we do not pay camera math every frame unless we choose to
	-- - \\ why: camera projection tests are non-trivial; decoupling lets you tune "popping" vs cost
	-- - \\ why: visibility changes slower than render frames, so sampling visibility at 10-20hz is usually enough
	CullHz: number,

	-- - \\ distance cap is a simple safety valve for giant maps
	-- - \\ why: world to viewport checks + cframe writes scale with active count; a hard radius guarantees an upper bound
	-- - \\ why: far-away particles provide almost no value, so they are the first to cheap-out
	MaxDrawDistance: number,

	-- - \\ stored here so workers do not need to call workspace gravity
	-- - \\ why: keeping config in the message avoids workers reaching into services (less overhead, fewer cross-context calls)
	-- - \\ why: if gravity changes per-map, the client can push the new value without patching workers
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
	-- - \\ why: initial landing plane can become invalid when geometry changes (destructible, moving parts, streamed chunks)
	-- - \\ why: refreshing support prevents "hovering decals" and lets gravity re-take control naturally
	SupportHz: number,
	SupportUp: number,
	SupportDown: number,
	SupportEps: number,

	-- - \\ sweeps are budgeted so fast particles do not tunnel but we also do not spike the frame
	-- - \\ why: continuous collision detection (sweeps) is the correct fix for tunneling, but it is also the most expensive
	-- - \\ why: budget caps prevent worst-case "spray into wall" from doing thousands of raycasts in a single tick
	WallPad: number,
	WallMaxLen: number,

	-- - \\ abs of ny below this means we treat the surface as a side wall
	-- - \\ why: floors/ground are handled by support logic; walls need different response (slide / stop / bounce)
	-- - \\ why: using ny is a cheap classifier instead of material tags or complicated normals logic
	WallNormalYMax: number,

	-- - \\ ny at or below negative this means we treat it as a ceiling while airborne
	-- - \\ why: ceilings are special because you can stick under them visually if you only do floor support checks
	-- - \\ why: only apply when airborne because grounded particles should already be stable and not bouncing
	CeilingNormalYMin: number,

	-- - \\ bounce is a simple visual cue and also stops particles from sticking under ceilings
	-- - \\ value is meant to be between zero and one
	-- - \\ why: a tiny bounce reads like splash energy and prevents the "glued to ceiling" artifact
	-- - \\ why: keeping it [0..1] makes tuning predictable (0 = dead stop, 1 = full reflect-ish)
	CeilingBounce: number,

	-- - \\ used for spawn occupancy checks and sweep stop distance so we do not clip through thin stuff
	-- - \\ why: we treat particles like spheres for cheap collision; radius is the one parameter that keeps it consistent
	-- - \\ why: both overlap tests and sweep stop distance depend on the same notion of "how big is a splat blob"
	CollisionRadius: number,

	-- - \\ hard caps for raycasts per tick so worst case stays predictable
	-- - \\ why: you cannot allow unbounded raycasts on client; it will eventually hitch on some machine or some map
	-- - \\ why: separate caps because airborne moves fastest (needs more ccd), sliding is slower (can be cheaper)
	MaxAirRaycasts: number,
	MaxSlideRaycasts: number,
}

type BloodSystem = {
	Config: Config,
	Initialized: boolean,

	-- - \\ sharedtable is the worker contract so it stays flat and array based
	-- - \\ why: struct-of-arrays is cache friendly + avoids allocating per-particle tables every splash
	-- - \\ why: workers can do tight numeric loops without table churn or metamethod surprises
	_state: any,

	-- - \\ parts are owned only by the client thread so workers never touch instances
	-- - \\ why: instances are not safe across parallel contexts and can force engine synchronization
	-- - \\ why: keeping instance ownership here avoids unpredictable costs and thread violations
	_parts: { BasePart },

	-- - \\ cached visibility so we avoid writing cframes for offscreen items
	-- - \\ why: cframe assignment is one of the most expensive per-item operations on the client
	-- - \\ why: if it's not on screen, moving it buys you nothing but cost
	_visible: { [number]: boolean },

	-- - \\ dense active list keeps work proportional to what is alive not pool size
	-- - \\ why: pool size is a capacity; active count is the true workload
	-- - \\ why: iterating 800 every tick is wasteful if only 40 are alive
	_activeList: { number },

	-- - \\ position map enables fast swap remove without searching
	-- - \\ why: removing from the middle of an array is expensive unless you swap-remove
	-- - \\ why: map gives O(1) "where is idx in activeList" so reclamation stays cheap under mass death
	_activePos: { [number]: number },

	-- - \\ free stack gives constant time allocation under spam
	-- - \\ why: this is the "drop particles instead of hitch" mechanism; allocation never searches for holes
	-- - \\ why: stack pop is stable and fast compared to scanning for inactive indices
	_free: { number },

	_folder: Folder?,

	-- - \\ cloned actors let us spread stepping across parallel contexts
	-- - \\ why: one actor is still one lua context; multiple actors gives parallel throughput
	-- - \\ why: we choose to parallelize the numeric integration instead of the rendering
	_actors: { Actor },

	_conn: RBXScriptConnection?,

	-- - \\ cached ray params so we do not allocate or rebuild filters every tick
	-- - \\ why: creating params / rebuilding filters causes allocations and can trigger gc
	-- - \\ why: raycasts are frequent, so we prebuild the params and reuse
	_map: Instance?,
	_rp: RaycastParams?,
	_op: OverlapParams?,
}

local ClientBloodSystem = {}
ClientBloodSystem.__index = ClientBloodSystem

-- - \\ key must match the worker key or the client and worker will talk past each other
-- - \\ why: sharedtable registry is basically a global dictionary; mismatch means worker reads a different table and nothing updates
-- - \\ why: versioning in the key is a cheap guard so old workers don't accidentally interpret new schema
local STATE_KEY = "_BLOOD_STATE_V1"

local function nowCamera(): Camera?
	-- - \\ camera can be replaced by respawn or camera scripts so we always fetch current
	-- - \\ why: storing camera once can go stale and cause nil access or parenting into the wrong object
	-- - \\ why: using currentcamera each time is cheap compared to debugging broken camera swaps
	return Workspace.CurrentCamera
end

local function ensureFolder(sys: BloodSystem): Folder
	-- - \\ keep all instances under one folder so cleanup is easy and the tree stays tidy
	-- - \\ why: pooling means lots of parts exist for the entire session; grouping prevents workspace clutter
	-- - \\ parenting under camera makes it obvious this is client only and avoids cluttering workspace
	-- - \\ why: camera is a natural "client visual container"; it won't replicate and signals ownership clearly
	if sys._folder and sys._folder.Parent ~= nil then
		return sys._folder
	end

	local f = Instance.new("Folder")
	f.Name = sys.Config.FolderName

	local cam = nowCamera()
	if cam then
		f.Parent = cam
	else
		-- - \\ camera can be nil briefly during load so we still allow init to complete
		-- - \\ why: blocking init on camera availability can deadlock effects during early load / respawn windows
		-- - \\ why: we prefer a safe fallback parent and later reparent when the camera exists
		f.Parent = Workspace
	end

	sys._folder = f
	return f
end

local function makePartFromTemplate(sys: BloodSystem): BasePart
	-- - \\ pooling is the main lever for smoothness since clone destroy churn causes gc hitches
	-- - \\ why: the expensive part is not just clone; it's also property replication inside engine + eventual gc pressure
	-- - \\ why: by creating once and reusing, we pay cost upfront and keep runtime stable
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

	-- - \\ visuals only so we shut off physics and queries to avoid extra cost and weird interactions
	-- - \\ why: physics solver work (contacts, broadphase) is wasted if we are manually simming + anchoring
	-- - \\ why: queries/touch events can create hidden overhead and unexpected gameplay triggers
	p.Anchored = true
	p.CanCollide = false
	p.CanQuery = false
	p.CanTouch = false
	p.CastShadow = false

	-- - \\ start hidden so there is never a flash on creation or reuse
	-- - \\ why: when parts are created/reparented, you can get a single-frame visibility glitch if transparency isn't set first
	-- - \\ why: on reuse, you don't want the old cframe to flash before you place it
	p.Transparency = 1
	p.Name = "Blood"

	return p
end

-- - \\ minimal schema guard so stale sharedtables do not crash the render loop
-- - \\ why: sharedtable registry can hold an older schema after hot reload; accessing missing arrays would hard error
-- - \\ why: we do the validation once at init instead of paying checks inside every tight loop
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

local function isSharedArray(t: any, poolSize: number): boolean
	-- - \\ accept plain tables for quick tests but sharedtable is what workers are tuned for
	-- - \\ why: this makes the module easier to unit test without actors / sharedtable registry
	-- - \\ why: production path is sharedtable because it has better cross-context semantics and performance expectations
	local tt = typeof(t)
	if tt ~= "SharedTable" and tt ~= "table" then
		return false
	end

	-- - \\ cheap check that the array is indexable up to the pool size without scanning it
	-- - \\ why: scanning every element is O(n) and can cost during init; we only need to know "does it look sized"
	-- - \\ why: checking last index catches common mismatch cases without doing work proportional to pool size
	return t[poolSize] ~= nil
end

local function stateLooksValid(st: any, poolSize: number): boolean
	-- - \\ if pool sizes mismatch then part index mapping breaks immediately
	-- - \\ why: part i is assumed to correspond to state arrays at i; mismatched size would index nil or wrong particle
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
	-- - \\ struct of arrays avoids per particle tables and keeps mutation cheap for workers
	-- - \\ why: per-particle tables cause allocator churn and poor cache locality
	-- - \\ why: arrays let workers do contiguous numeric access patterns (fewer hash lookups)
	local t = SharedTable.new()
	for i = 1, poolSize do
		t[i] = initValue
	end
	return t
end

local function buildState(poolSize: number): any
	-- - \\ one place to define the schema so it does not drift between client and worker
	-- - \\ why: schema drift is the #1 cause of "nothing moves" bugs in shared memory setups
	-- - \\ why: keeping it centralized ensures adding a field is a single edit, not a multi-file hunt
	local st = SharedTable.new()

	st.poolSize = poolSize

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
	-- - \\ reusing a valid existing state makes hot reload less jarring and avoids instant resets
	-- - \\ why: during live edit, resetting the pool can cause visual popping + broken indices mid-frame
	-- - \\ why: if schema matches, reuse is strictly cheaper and more stable
	local ok, existing = pcall(function()
		return SharedTableRegistry:GetSharedTable(STATE_KEY)
	end)

	if ok and existing and stateLooksValid(existing, poolSize) then
		return existing
	end

	local st = buildState(poolSize)
	-- - \\ we set it only when we know it matches our schema
	-- - \\ why: registry is shared; if you set partial/invalid state, all workers will read garbage
	SharedTableRegistry:SetSharedTable(STATE_KEY, st)
	return st
end

local function computeSurfaceBasis(nx: number, ny: number, nz: number): (Vector3, Vector3)
	-- - \\ grounded particles need a stable basis or they can jitter when normals are borderline
	-- - \\ why: if you build orientation from an unstable cross product, tiny normal noise causes large yaw flips
	-- - \\ handle degenerate normals and near parallel cases so we never end up with zero vectors
	-- - \\ why: a near-zero vector normalized becomes nan/inf which will explode cframes and poison the render path
	local up = Vector3.new(nx, ny, nz)
	if up.Magnitude < 1e-6 then
		up = Vector3.new(0, 1, 0)
	else
		up = up.Unit
	end

	local ref = Vector3.new(0, 1, 0)
	local right = up:Cross(ref)
	if right.Magnitude < 1e-6 then
		-- - \\ if up is basically parallel to world up, we pick a different reference axis
		-- - \\ why: cross of parallel vectors is ~0, so we need a fallback axis to build a valid basis
		ref = Vector3.new(1, 0, 0)
		right = up:Cross(ref)
	end

	right = right.Unit
	return right, up
end

local function makeSurfaceCFrame(px: number, py: number, pz: number, nx: number, ny: number, nz: number): CFrame
	-- - \\ frommatrix is consistent and avoids lookat ambiguity on shallow slopes
	-- - \\ why: lookat can twist unpredictably because "forward" is underconstrained when you only know an up normal
	-- - \\ why: frommatrix lets us explicitly define right/up/back for deterministic orientation
	local pos = Vector3.new(px, py, pz)
	local right, up = computeSurfaceBasis(nx, ny, nz)
	local back = right:Cross(up)
	return CFrame.fromMatrix(pos, right, up, back)
end

local function defaultConfig(): Config
	-- - \\ defaults aim for nice visuals while keeping worst case work bounded
	-- - \\ why: defaults should be "safe" on mid-tier hardware and big maps; people can scale up later
	-- - \\ why: the limiting factors here are raycasts and cframe writes, so budgets are set accordingly
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

		-- - \\ support checks happen often so we keep them lean
		-- - \\ why: support checks are frequent and can dominate raycast count; low-cost math + single ray is the sweet spot
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

		-- - \\ budgets keep airborne reliable and cap slide sweeps hard
		-- - \\ why: airborne has highest speed so it gets more ccd budget; sliding gets less because it should be slower and mostly constrained
		MaxAirRaycasts = 220,
		MaxSlideRaycasts = 120,
	}
end

function ClientBloodSystem.new(config: Config?): BloodSystem
	-- - \\ copy config once so callers cannot mutate tuning mid sim and cause hard to debug behavior
	-- - \\ why: shared config mutation mid-step can desync client vs worker assumptions (dt, damping, etc)
	-- - \\ why: freezing config also keeps profiling consistent because parameters don't drift during runtime
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
	-- - \\ cloning locally keeps deployment simple and guarantees sendmessage targets local actors
	-- - \\ why: sendmessage only works if the actor exists in this client environment; cloning makes it deterministic
	-- - \\ why: it also decouples the effect from server replication and avoids ordering issues
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
	-- - \\ stack pop keeps allocation constant time under spam
	-- - \\ why: constant time allocation is the only way to guarantee splash spam doesn't become an O(n) stall
	-- - \\ why: when empty, we return nil and the caller drops particles (the intentional safety valve)
	local n = #sys._free
	if n == 0 then
		return nil
	end

	local idx = sys._free[n]
	sys._free[n] = nil
	return idx
end

local function resolveSpawn(sys: BloodSystem, start: Vector3): Vector3
	-- - \\ starting inside geometry makes everything worse so we try a cheap overlap based nudge
	-- - \\ why: if you spawn inside a wall, you immediately trigger sweeps/support flips and spend extra raycasts recovering
	-- - \\ why: overlap nudge is cheaper than multi-ray "find nearest empty" and is good enough for visuals
	local op = sys._op
	if not op then
		return start
	end

	local r = math.max(0.15, sys.Config.CollisionRadius * 0.90)

	local function occupied(p: Vector3): boolean
		-- - \\ maxparts is one so we get a quick any hit test
		-- - \\ why: we only need boolean occupancy, not the full set; limiting maxparts keeps it fast and predictable
		local parts = Workspace:GetPartBoundsInRadius(p, r, op)
		return parts[1] ~= nil
	end

	if not occupied(start) then
		return start
	end

	-- - \\ small search that avoids multi ray solutions
	-- - \\ why: this is intentionally bounded (few offsets, few steps) so it can't blow up cost in degenerate maps
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

	-- - \\ downward bias usually looks less weird than shoving sideways
	-- - \\ why: pushing sideways can visibly teleport out of an impact point; dropping down reads like gravity
	return start - Vector3.new(0, sys.Config.CollisionRadius, 0)
end

function ClientBloodSystem:Init()
	-- - \\ lazy init so we only pay pool allocation when the effect is actually used
	-- - \\ why: pool creation is an upfront cost; delaying it keeps initial load/menus smoother
	-- - \\ why: also supports "module exists but never used" without wasting memory
	if self.Initialized then
		return
	end

	assert(RunService:IsClient(), "ClientBloodSystem must run on the client")

	-- - \\ state creation is centralized here so the worker can assume the arrays exist
	-- - \\ why: workers should run hot; they shouldn't build schema or defend nils every step
	self._state = getOrCreateState(self.Config.PoolSize)

	-- - \\ cache map and params so per tick work stays mostly math and not allocations
	-- - \\ why: raycast/overlap params are objects; creating them repeatedly causes allocations and gc pressure
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
	-- - \\ why: instance churn is the main hitch source; stable pool means stable frametime
	local folder = ensureFolder(self)
	table.clear(self._parts)
	table.clear(self._free)

	for i = 1, self.Config.PoolSize do
		local p = makePartFromTemplate(self)
		p.Parent = folder

		-- - \\ park far away while hidden so reuse never flashes at origin
		-- - \\ why: some engines/clients can show a part for a frame during parenting; parking it prevents a visible "pop"
		p.CFrame = CFrame.new(0, -1e6, 0)

		self._parts[i] = p
		self._visible[i] = false
		self._free[i] = i

		-- - \\ keep arrays non nil so workers never branch on missing fields
		-- - \\ why: pre-filling also defends against partial state if something interrupts init
		self._state.active[i] = 0
		self._state.phase[i] = 0
		self._state.age[i] = 0
		self._state.support[i] = 0
		self._state.wall[i] = 0
		self._state.wallt[i] = 0
	end

	-- - \\ actors do the heavy stepping so the main thread can stay focused on rendering
	-- - \\ why: keeps cframe writes + visibility logic responsive under heavy blood spam
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
		-- - \\ cframe writes are the hot path so we only do them when actually visible
		-- - \\ why: visibility is the gating signal for "should we spend cframe work"
		-- - \\ why: we also update transparency here so hidden parts don't cost fillrate or draw calls
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
					-- - \\ margin reduces popping when the camera jitters
					-- - \\ why: camera shake / headbob can flicker edges; margin gives hysteresis without state machines
					if v.X >= -50 and v.Y >= -50 and v.X <= (vpSize.X + 50) and v.Y <= (vpSize.Y + 50) then
						vis = true
					end
				end
			end

			if self._visible[idx] ~= vis then
				self._visible[idx] = vis
				self._parts[idx].Transparency = if vis then 0 else 1
			end
		end
	end

	local function applyVisuals()
		-- - \\ reclamation happens here because the client owns the pool and the active list
		-- - \\ why: only the client should decide "free vs active" because it owns parts + the free stack
		-- - \\ the worker only flips active to zero when it is done
		-- - \\ why: worker finishing is a single bit write; client does the expensive bookkeeping and instance hiding
		for i = #self._activeList, 1, -1 do
			local idx = self._activeList[i]

			if self._state.active[idx] == 0 then
				local pos = self._activePos[idx]
				if pos then
					-- - \\ swap remove keeps removal fast even when a lot die at once
					-- - \\ why: we avoid table.remove shifting N elements, which spikes when many particles expire together
					local lastIdx = self._activeList[#self._activeList]
					self._activeList[pos] = lastIdx
					self._activePos[lastIdx] = pos
					self._activeList[#self._activeList] = nil
					self._activePos[idx] = nil
				end

				-- - \\ clear flags so the next reuse does not inherit stale contact state
				-- - \\ why: if you reuse an index with old wall/support flags, the worker may instantly resolve as if colliding
				self._state.wall[idx] = 0
				self._state.wallt[idx] = 0
				self._state.support[idx] = 0

				local part = self._parts[idx]
				part.Transparency = 1
				part.CFrame = CFrame.new(0, -1e6, 0)
				self._visible[idx] = false

				-- - \\ recycle index back to the pool
				-- - \\ why: returning to free stack is what keeps future spawn O(1)
				self._free[#self._free + 1] = idx
			else
				-- - \\ skip cframe writes when hidden because that is where the main thread cost piles up
				-- - \\ why: worker continues simming so state remains correct, but we avoid paying render-side cost
				if self._visible[idx] then
					local px = self._state.px[idx]
					local py = self._state.py[idx]
					local pz = self._state.pz[idx]

					local ay = self._state.ay[idx]
					local phase = self._state.phase[idx]

					local cf: CFrame
					if phase == 1 then
						-- - \\ airborne uses free rotation so it reads like spray not stickers
						-- - \\ why: airborne blobs are "volumetric" visually; arbitrary rotation reads like droplets tumbling
						local ax = self._state.ax[idx]
						local az = self._state.az[idx]
						cf = CFrame.new(px, py, pz) * CFrame.Angles(ax, ay, az)
					else
						-- - \\ grounded aligns to the surface normal so it sits cleanly on slopes
						-- - \\ why: once landed, viewers expect it to conform to the surface, otherwise it floats/clips
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
		-- - \\ split indices evenly across actors so one worker does not become the bottleneck
		-- - \\ why: even distribution prevents "one hot worker" stalling while others idle
		-- - \\ why: balancing is more important than perfect chunking because particle cost varies by collisions
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

			-- - \\ small slice allocation scales with worker count not particle count
			-- - \\ why: we allocate one slice per worker per dispatch; that is bounded by actor count (small)
			-- - \\ why: sending a contiguous slice avoids sending huge tables or per-particle messages
			local slice = table.create(endPos - startPos + 1)
			local s = 1
			for j = startPos, endPos do
				slice[s] = list[j]
				s += 1
			end

			-- - \\ send only what the worker needs so config cannot drift through shared state
			-- - \\ why: config in message is explicit and versionable; worker doesn't depend on shared mutable config
			-- - \\ why: less sharedtable traffic (only state arrays mutate), fewer accidental cross-context reads
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
		-- - \\ support refresh is what lets splats fall off edges instead of hovering on an old plane
		-- - \\ why: the worker may be using last-known plane normal/point; if the particle drifts over an edge, it must lose support
		-- - \\ why: we do this on client so workers don't each raycast (would multiply raycast cost by worker count)
		local rpLocal = self._rp :: RaycastParams

		for i = 1, #list do
			local idx = list[i]
			local phase = self._state.phase[idx]

			-- - \\ dead and fully resting do not need support work so we skip to save raycasts
			-- - \\ why: phase 0 means inactive, phase 3 means "resting" (basically stopped), so support updates give little value
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

					-- - \\ signed separation is a cheap stable way to decide supported without flicker
					-- - \\ why: projecting distance onto normal gives a consistent "how far above plane" metric
					-- - \\ why: eps creates hysteresis so small numeric noise doesn't toggle support every tick
					self._state.support[idx] = if sep <= (self.Config.SurfaceOffset + self.Config.SupportEps) then 1 else 0
				else
					self._state.support[idx] = 0
				end
			end
		end
	end

	local function updateWallsBudgeted(list: { number }, dt: number)
		-- - \\ continuous sweeps prevent tunneling but can get expensive
		-- - \\ why: discrete collision checks miss thin walls at high speed; sweeps use motion direction to catch first contact
		-- - \\ budgets keep worst case predictable and we prioritize airborne first since it moves fastest
		-- - \\ why: airborne is where tunneling looks the worst (droplets pass through walls); sliding is slower and more forgiving
		local rpLocal = self._rp :: RaycastParams

		local radius = self.Config.CollisionRadius
		local pad = self.Config.WallPad
		local maxLen = self.Config.WallMaxLen
		local nyMax = self.Config.WallNormalYMax
		local ceilMin = self.Config.CeilingNormalYMin

		local function tryOne(idx: number, phase: number): boolean
			-- - \\ do not redo a sweep if one already hit this frame
			-- - \\ why: multiple raycasts for the same particle in the same frame is pure waste; the worker will resolve based on stored toi
			if self._state.wall[idx] ~= 0 then
				return false
			end

			local vx = self._state.vx[idx]
			local vy = self._state.vy[idx]
			local vz = self._state.vz[idx]

			local sp2 = vx * vx + vy * vy + vz * vz
			if sp2 <= 0.0025 then
				-- - \\ low speed does not tunnel so we save the raycast
				-- - \\ why: raycasts should be reserved for fast movers; slow movers can be handled by simple support logic
				return false
			end

			local sp = math.sqrt(sp2)
			local ux = vx / sp
			local uy = vy / sp
			local uz = vz / sp

			local len = sp * dt + radius + pad
			if len > maxLen then
				-- - \\ hard cap stops one particle from firing huge rays into the map
				-- - \\ why: if dt spikes or velocity is high, len could become enormous and cost more (and hit unintended geometry)
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

			-- - \\ floors are handled by support logic so we only care about side walls and ceilings here
			-- - \\ why: treating floors here would duplicate support work and cause double-resolve jitter
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

			-- - \\ store contact and toi so the worker resolves without doing its own raycasts
			-- - \\ why: one raycast on the client feeds all workers; if each worker raycasted, budgets would multiply
			-- - \\ why: storing toi lets the worker integrate to contact point deterministically
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
		-- - \\ renderstepped keeps visuals tight and we still decouple stepping with accumulators
		-- - \\ why: renderstepped is synced to camera; writing cframes here minimizes perceived latency
		-- - \\ why: accumulators prevent "do everything every frame" and make stepping frequency deterministic
		local cam = nowCamera()
		if cam and self._folder and self._folder.Parent ~= cam then
			-- - \\ camera swaps happen so we reparent instead of rebuilding the pool
			-- - \\ why: rebuilding would re-clone parts and spike; reparent is cheap and preserves pooled instances
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
			-- - \\ partitioning means we spend raycasts and high rate only on visible particles
			-- - \\ why: visibility is the best proxy for "does the player benefit from this cost"
			-- - \\ why: resting particles are almost static; they can be updated rarely without visual loss
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
				-- - \\ clamp dt so one hitch does not turn into a huge catch up step
				-- - \\ why: big dt makes particles teleport and makes sweeps long (more likely to hit odd stuff)
				-- - \\ why: clamping stabilizes simulation and keeps ray lengths bounded even if the game hitches
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
				-- - \\ hidden still steps so it does not freeze then pop back in wrong
				-- - \\ why: if hidden particles pause, when they re-enter view they will appear "stuck in the past"
				-- - \\ why: lower hz keeps correctness without paying full rate
				local stepDt = math.min(physAccumHid, 0.1)
				physAccumHid = 0
				dispatchPhysics(stepDt, hidList)
			end

			if doRest then
				-- - \\ resting is low motion so low rate is fine and saves a lot of work
				-- - \\ why: most cost comes from collision + integration; resting does neither much, so low rate is nearly free
				local stepDt = math.min(physAccumRest, 0.1)
				physAccumRest = 0
				dispatchPhysics(stepDt, restList)
			end
		end
	end)
end

function ClientBloodSystem:Splash(at: Vector3 | BasePart)
	-- - \\ callers should not need to care about init order so we self init on first use
	-- - \\ why: splash is the "public api"; making it safe removes ordering bugs in weapons / effects code
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
		-- - \\ if we are saturated we drop particles instead of stalling the client
		-- - \\ why: when pool is empty, the only alternative is realloc/resize (expensive) or searching (expensive)
		-- - \\ why: dropping is the chosen failure mode because it's visually acceptable and performance safe
		local idx = popFree(self)
		if not idx then
			break
		end

		-- - \\ tiny jitter makes repeated splashes look less copy paste and helps avoid identical overlaps
		-- - \\ why: identical positions cause identical ray hits and can stack visually in an unnatural way
		-- - \\ why: jitter also helps occupancy checks find a nearby free spot instead of hammering the same blocked center
		local jitter = Vector3.new(
			(math.random() - 0.5) * 0.25,
			(math.random() - 0.5) * 0.10,
			(math.random() - 0.5) * 0.25
		)

		-- - \\ keep starts out of geometry so collision recovery does not waste a bunch of work
		-- - \\ why: starting inside means immediate wall hits, more sweeps, and ugly "pop-out" artifacts
		local start = resolveSpawn(self, pos + jitter)

		-- - \\ seed an initial support plane so the first few frames look consistent
		-- - \\ why: if you wait for the support refresh tick, the particle might render with default normal (0,1,0) and pop orientation later
		-- - \\ why: one ray here is cheap and improves "first frame" stability a lot
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

		-- - \\ random spray is cheap variety without needing per weapon tuning
		-- - \\ why: you get "organic" variation from simple uniform randomness without authoring curves per gun
		-- - \\ why: the min/max ranges keep it bounded so droplets don't fly across the entire map
		local theta = math.random() * math.pi * 2
		local out = self.Config.ArchOutMin + (self.Config.ArchOutMax - self.Config.ArchOutMin) * math.random()
		local up = self.Config.ArchUpMin + (self.Config.ArchUpMax - self.Config.ArchUpMin) * math.random()

		local vx = math.cos(theta) * out
		local vz = math.sin(theta) * out
		local vy = up

		-- - \\ if we still overlap after resolve then bias downward so we do not pop out sideways
		-- - \\ why: sideways pop reads like teleport and can push through walls; downward bias reads like "it splatted into something"
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

		-- - \\ spin sells the spray and damping handles the settle after landing
		-- - \\ why: spin provides micro-motion so airborne looks energetic
		-- - \\ why: damping prevents infinite spin and gives a natural "settle" once grounded
		local avx = (math.random() - 0.5) * 30
		local avy = (math.random() - 0.5) * 44
		local avz = (math.random() - 0.5) * 30

		self._state.active[idx] = 1
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

		-- - \\ random start orientation keeps repeated splats from looking identical
		-- - \\ why: if all splats share the same rotation, players notice tiling/patterning immediately
		-- - \\ why: random orientation is essentially free variety (just 3 numbers) with big visual payoff
		self._state.ax[idx] = math.random() * math.pi * 2
		self._state.ay[idx] = math.random() * math.pi * 2
		self._state.az[idx] = math.random() * math.pi * 2

		self._state.avx[idx] = avx
		self._state.avy[idx] = avy
		self._state.avz[idx] = avz

		self._state.support[idx] = 0
		self._state.wall[idx] = 0
		self._state.wallt[idx] = 0

		-- - \\ local tracking keeps iteration fast and lets us do swap removes without touching the pool
		-- - \\ why: activeList is the tight loop list; pool is capacity-only and should not be scanned
		-- - \\ why: swap remove needs a position map to stay O(1)
		self._activeList[#self._activeList + 1] = idx
		self._activePos[idx] = #self._activeList

		self._parts[idx].Parent = folder
		self._parts[idx].Transparency = 1
		self._visible[idx] = false
	end
end

function ClientBloodSystem:Destroy()
	-- - \\ explicit teardown avoids leaked connections and pooled instances during mode switches or reloads
	-- - \\ why: renderstepped connections keep running even if you think the system is gone; leaks cause hidden perf drain
	-- - \\ why: pooled parts stick around unless destroyed; during map swaps you want clean reset and memory release
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
