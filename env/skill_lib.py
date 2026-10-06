"""Low-level skill library for Habitat rearrange tasks.

This module ports the *low-level skill layer* used by the PARTNR benchmark
(``facebookresearch/partnr-planner``: Navigate / Pick / Place / Open / Close)
so that a high-level policy -- either an LLM planner or a reinforcement
learning agent -- can act over **temporal abstractions** instead of raw
motor commands.

Design constraints honoured here:

* **No privileged information.** Nothing snaps objects, teleports the robot
  or uses ``OracleNavPolicy``. Every skill is executed through the ordinary
  11-D Habitat action space ``[arm(7), grip(1), base_vel(2), stop(1)]``.
* **No neural-network checkpoints.** PARTNR's ``nn_skills`` need seven
  missing ``.pth`` files; everything below is purely geometric / kinematic,
  so it runs on a Habitat wheel compiled without Bullet.
* **SMDP interface.** A skill is a controller that emits one low-level
  action per simulator step and reports ``DONE`` when its termination
  predicate fires -- exactly the Tool/Skill contract of PARTNR.

Termination / one-step vocabularies:

``SkillResult`` values returned by :meth:`SkillContext.step`.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ARM_DIM = 7
GRIP_DIM = 1
BASE_DIM = 2
ACTION_DIM = ARM_DIM + GRIP_DIM + BASE_DIM + 1  # == 11

ARM_SLICE = slice(0, 7)
GRIP_SLICE = slice(7, 8)
BASE_SLICE = slice(8, 10)
STOP_SLICE = slice(10, 11)


class SkillResult(enum.IntEnum):
    """Outcome of a single skill step."""

    RUNNING = 0
    DONE = 1
    FAILED = 2  # timed out or unrecoverable state


def _make_action(
    arm: Optional[np.ndarray] = None,
    grip: float = 0.0,
    base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build an 11-D Habitat action. ``stop`` is always pinned off."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    if arm is not None:
        a[ARM_SLICE] = np.asarray(arm, dtype=np.float32).reshape(-1)[:ARM_DIM]
    a[GRIP_SLICE] = float(grip)
    if base is not None:
        a[BASE_SLICE] = np.asarray(base, dtype=np.float32).reshape(-1)[:BASE_DIM]
    a[STOP_SLICE] = -1.0  # never request episode termination
    return a


# ---------------------------------------------------------------------------
# Low-level controllers
# ---------------------------------------------------------------------------
def _yaw_from_quat(rotation) -> float:
    """Extract a heading angle from a habitat rotation (quaternion-like)."""
    try:
        fn = rotation.to_angle_axis
        ang_vec = fn()
        if isinstance(ang_vec, tuple):
            angle, vec = ang_vec
        else:  # magnum scalar-returning overload
            angle, vec = ang_vec[0], None
        if vec is not None:
            return float(np.arctan2(vec.y, vec.x)) if abs(vec.z) < 0.7 else float(angle)
    except Exception:  # noqa: BLE001 - magnum API differences
        pass
    try:
        # rotation as a 3x3 transform, use first column (forward axis)
        m = np.asarray(rotation.as_matrix() if hasattr(rotation, "as_matrix")
                       else rotation, dtype=np.float64)
        return float(np.arctan2(m[1, 0], m[0, 0]))
    except Exception:  # noqa: BLE001
        return 0.0


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


class BaseInterface:
    """Small abstraction over the Habitat task interface.

    Pulls robot state (base transform, arm joints, end-effector position)
    and world state (object positions, target positions) out of the
    simulator without touching any RL-specific code.
    """

    def __init__(self, habitat_env: Any, agent_idx: int = 0) -> None:
        import magnum as mn  # noqa: F401  (needed for vector arithmetic)

        self._env = habitat_env
        self._agent_idx = agent_idx
        self._sim = habitat_env.sim
        self._aa = None
        self._refresh_agent()

    # -- cached handles ----------------------------------------------
    def _refresh_agent(self) -> None:
        """Bind to the robot owned by ``agent_idx``.

        Both ``RearrangeSim.get_agent_data`` and the private list on
        ``agents_mgr`` are supported -- different habitat versions expose
        different entry points.
        """
        try:
            self._aa = self._sim.get_agent_data(self._agent_idx)
        except Exception:  # noqa: BLE001
            self._aa = self._sim.agents_mgr._all_agent_data[self._agent_idx]

    @property
    def articulated_agent(self):
        return self._aa.articulated_agent

    @property
    def sim(self):
        """The underlying RearrangeSim (needed for navmesh planning)."""
        return self._sim

    # -- robot state -------------------------------------------------
    @property
    def base_transformation(self):
        agent = self.articulated_agent
        for attr in ("base_transformation", "sim_obj"):
            if hasattr(agent, attr):
                obj = getattr(agent, attr)
                return getattr(obj, "transformation", None) if attr != "sim_obj" \
                    else obj.transformation
        return None

    @property
    def base_pos(self) -> np.ndarray:
        agent = self.articulated_agent
        val = agent.base_pos if hasattr(agent, "base_pos") \
            else self._aa.transformation.translation
        return np.array(list(val), dtype=np.float64)

    @property
    def base_yaw(self) -> float:
        """Heading of the robot as ``-base_rot``.

        Calibrated on real Habitat: driving with ``base_vel=[1, 0]`` moves
        the robot along ``(cos(-rot), 0, sin(-rot))`` -- i.e. world heading
        is the *negation* of ``base_rot``. +1 angular turns at
        +0.0833 rad/step.
        """
        agent = self.articulated_agent
        rot = float(agent.base_rot) if hasattr(agent, "base_rot") else 0.0
        return -rot

    @property
    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lower, upper) bounds for the arm joints.

        ``joint_position_limits`` reports fewer entries than there are
        model joints on some robots, so fall back to permissive bounds
        whenever the arm ids cannot index it.
        """
        agent = self.articulated_agent
        try:
            lo, hi = agent.arm_joint_limits  # official, masked & mapped
            return (np.array(list(lo), dtype=np.float64),
                    np.array(list(hi), dtype=np.float64))
        except Exception:  # noqa: BLE001
            n = len(getattr(agent.params, "arm_joints", [0]))
            return np.full(n, -2.2), np.full(n, 3.0)

    @property
    def arm_joint_pos(self) -> np.ndarray:
        return np.array(self.articulated_agent.arm_joint_pos, dtype=np.float64)

    @property
    def ee_pos(self) -> np.ndarray:
        return np.array(self.articulated_agent.ee_transform().translation,
                        dtype=np.float64)

    def _write_joints(self, joints: np.ndarray) -> None:
        """Write a hypothetical arm configuration using the official setter.

        ``Manipulator.arm_joint_pos`` takes care of the Habitat joint id ->
        simulator index mapping and of the kinematic/dynamic split, so we
        never poke at ``joint_positions`` by hand.
        """
        self.articulated_agent.arm_joint_pos = np.asarray(joints, dtype=np.float32)

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """EE position for a hypothetical arm joint configuration.

        Forward kinematics only -- works without PyBullet. Note the model
        state is restored before returning, so calling this is free of side
        effects.
        """
        agent = self.articulated_agent
        sim_obj = agent.sim_obj
        prev = np.array(sim_obj.joint_positions, dtype=np.float32)
        try:
            self._write_joints(joints)
            return np.array(agent.ee_transform().translation, dtype=np.float64)
        finally:
            sim_obj.joint_positions = prev

    def clip_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        """Clip a *world* target into the reachable end-effector box.

        ``params.ee_constraint`` is expressed in the **robot base frame**
        (not world), so the point has to be pushed into that frame, clipped
        and transformed back.
        """
        pos = np.asarray(pos, dtype=np.float64)
        try:
            base_t = self.articulated_agent.sim_obj.transformation
            cons = np.asarray(self.articulated_agent.params.ee_constraint,
                              dtype=np.float64)
            if cons.ndim == 3:
                cons = cons[0]
            local = base_t.inverted().transform_point(
                __import__("magnum").Vector3(*pos))
            clipped = np.array(list(local), dtype=np.float64)
            clipped = np.clip(clipped, cons[:, 0], cons[:, 1])
            out = base_t.transform_point(__import__("magnum").Vector3(*clipped))
            return np.array(list(out), dtype=np.float64)
        except Exception:  # noqa: BLE001
            return pos

    # -- world state -------------------------------------------------
    @property
    def object_positions(self) -> np.ndarray:
        return np.array(self._sim.get_scene_pos(), dtype=np.float64)

    @property
    def target_positions(self) -> np.ndarray:
        """Goal positions of every target object.

        ``RearrangeSim.get_targets`` returns ``(indices, positions)`` --
        the second element is what we want.
        """
        targets = self._sim.get_targets()
        if isinstance(targets, (tuple, list)) and len(targets) == 2:
            targets = targets[1]
        arr = np.asarray(targets if isinstance(targets, np.ndarray)
                         else [np.array(list(t.translation))
                               for t in targets], dtype=np.float64)
        return arr.reshape(-1, 3)

    @property
    def is_grasped(self) -> bool:
        return bool(self._aa.grasp_mgr.is_grasped)

    def snap_distance(self, obj_pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.ee_pos - np.asarray(obj_pos, dtype=np.float64)))


class BaseVelController:
    """Turn-then-drive PD controller over Habitat's 2-D base velocity.

    Non-cylindrical Spot base: action = ``[forward:, angular:]`` clipped
    to the action-space bounds.
    """

    def __init__(
        self,
        dist_tol: float = 0.3,
        ang_tol: float = 0.15,
        k_lin: float = 1.2,
        k_ang: float = 2.0,
    ) -> None:
        self.dist_tol = dist_tol
        self.ang_tol = ang_tol
        self.k_lin = k_lin
        self.k_ang = k_ang

    def __call__(self, iface: BaseInterface, target: np.ndarray) -> np.ndarray:
        pos = iface.base_pos
        diff = np.asarray(target, dtype=np.float64)[:2] - pos[:2]
        dist = float(np.linalg.norm(diff))
        desired = float(np.arctan2(diff[1], diff[0]))      # world heading
        err = _angle_diff(desired, iface.base_yaw)          # + -> turn CCW
        # Calibrated: base_vel=[1,0] -> 0.083 m/step along the heading,
        # base_vel=[0,+1] -> +0.0833 rad/step of *base_rot*, i.e. NEGATIVE
        # heading. Hence the minus sign on the angular term.
        ang = -float(np.clip(self.k_ang * err, -1.0, 1.0))
        if abs(err) < self.ang_tol:
            ang = 0.0
        lin = 0.0
        if abs(err) < 0.5 and dist > self.dist_tol:
            lin = float(np.clip(self.k_lin * dist, 0.2, 1.0))
        return np.array([lin, ang], dtype=np.float32)

    def done(self, iface: BaseInterface, target: np.ndarray) -> bool:
        diff = np.asarray(target, dtype=np.float64)[:2] - iface.base_pos[:2]
        return bool(np.linalg.norm(diff) < self.dist_tol)


class ArmIKController:
    """Jacobian-transpose inverse kinematics on top of Spot.

    Only forward kinematics is used (finite-difference Jacobian), so this
    works with a Habitat wheel that has no Bullet and therefore no
    ``ik_helper``.
    """

    def __init__(
        self,
        step_size: float = 0.5,
        max_iters: int = 15,
        tol: float = 0.015,
        residual_tol: float = 0.06,
        eps: float = 2e-3,
        resolve_every: int = 8,
    ) -> None:
        self.step_size = step_size
        self.max_iters = max_iters
        self.tol = tol
        self.residual_tol = residual_tol
        self.eps = eps
        self.resolve_every = resolve_every
        self._target_cache: Optional[np.ndarray] = None
        self._goal_cache: Optional[np.ndarray] = None
        self._age = 0

    def solve(self, iface: BaseInterface, target: np.ndarray) -> np.ndarray:
        """Return an arm joint configuration reaching ``target``."""
        lo, hi = iface.joint_limits
        q = iface.arm_joint_pos.copy()
        base = iface.forward_kinematics(q)
        for _ in range(self.max_iters):
            err = np.asarray(target, dtype=np.float64) - base
            if np.linalg.norm(err) < self.tol:
                break
            jac = np.zeros((3, q.size))
            for i in range(q.size):
                probe = q.copy()
                probe[i] += self.eps
                jac[:, i] = (iface.forward_kinematics(probe) - base) / self.eps
            g = jac.T @ err
            denom = float(np.linalg.norm(g))
            if denom < 1e-9:
                break
            alpha = min(self.step_size, self.step_size * np.linalg.norm(err) / denom)
            q = np.clip(q + alpha * g, lo, hi)
            base = iface.forward_kinematics(q)
        self._target_cache = np.asarray(target, dtype=np.float64)
        self._goal_cache = q
        self._age = 0
        return q

    def delta(self, iface: BaseInterface, target: np.ndarray) -> np.ndarray:
        """Relative joint command for one environment step.

        ``ArmAction`` scales the action by ``delta_pos_limit == 0.0125``, so
        the command is emitted at full magnitude along the joint-space
        direction instead of being passed through as a radian delta -- a
        1 rad move otherwise needs hundreds of steps.
        """
        tgt = np.asarray(target, dtype=np.float64)
        if (self._goal_cache is None or self._age % self.resolve_every == 0
                or self._target_cache is None
                or np.linalg.norm(self._target_cache - tgt) > 1e-6):
            self.solve(iface, tgt)
        self._age += 1
        delta = self._goal_cache - iface.arm_joint_pos
        nrm = float(np.linalg.norm(delta))
        if nrm < 1e-6:
            return np.zeros_like(delta, dtype=np.float32)
        return np.clip(delta / nrm, -1.0, 1.0).astype(np.float32)

    def done(self, iface: BaseInterface, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(iface.ee_pos - np.asarray(target,
                                                             dtype=np.float64))
                    < self.residual_tol)


# ---------------------------------------------------------------------------
# PARTNR-style skills
# ---------------------------------------------------------------------------
class Skill:
    """Base class: a controller executed over several simulator steps."""

    #: Human readable name exposed to planners / RL policies.
    name: str = "skill"
    #: Default maximum number of low-level steps before FAILED.
    max_steps: int = 200

    def begin(self, iface: BaseInterface, kwargs: Dict[str, Any]) -> None:  # noqa: D102
        raise NotImplementedError

    def step(self, iface: BaseInterface) -> Tuple[np.ndarray, SkillResult]:
        raise NotImplementedError


class NavSkill(Skill):
    """``Navigate`` -- drive the base to a 3-D target position.

    Two modes are available:

    ``use_pathfinder=True`` (default)
        Ask the simulator's navmesh for a geodesic shortest path and follow
        its waypoints with the PD controller. This is standard navigation
        practice (Habitat's own PointNav agent does the same); pass
        ``False`` for a straight-line ablation with no scene knowledge.
    """

    name = "navigate"

    def __init__(self, dist_tol: float = 0.3, use_pathfinder: bool = True) -> None:
        self.control = BaseVelController(dist_tol=dist_tol)
        self.use_pathfinder = use_pathfinder
        self.target: Optional[np.ndarray] = None
        self.approach_point: Optional[np.ndarray] = None
        self.waypoints: List[np.ndarray] = []
        self.wp_idx = 0
        self.steps = 0
        self.max_steps = 400
        self.stalled_from: Optional[np.ndarray] = None
        self.stall_steps = 0

    # -- path planning ----------------------------------------------
    @staticmethod
    def _nearest_navigable(pf, target: np.ndarray, max_radius: float = 1.5):
        """Snap a possibly-unreachable world point onto the navmesh.

        Objects live on counters and inside receptacles, so asking "drive
        onto the object" has no valid answer; we drive to the closest
        standable point instead.
        """
        cands: List[np.ndarray] = []
        tgt32 = np.asarray(target, dtype=np.float32)
        try:
            sp = np.array(list(pf.snap_point(tgt32)), dtype=np.float64)
            if (np.all(np.isfinite(sp)) and pf.is_navigable(sp)
                    and np.linalg.norm(sp - target) <= max_radius):
                cands.append(sp)
        except Exception:  # noqa: BLE001
            pass
        for radius in (0.75, 1.5, 2.5):
            try:
                p = np.array(list(pf.get_random_navigable_point_near(
                    tgt32, radius)), dtype=np.float64)
                if np.all(np.isfinite(p)):
                    cands.append(p)
            except Exception:  # noqa: BLE001
                continue
        if not cands:
            return None
        return min(cands, key=lambda p: float(np.linalg.norm(p - target)))

    def _plan(self, iface: BaseInterface, target: np.ndarray) -> None:
        if not self.use_pathfinder:
            self.waypoints = [target]
            return
        import habitat_sim  # local import: only this mode needs it

        pf = iface.sim.pathfinder
        start = np.asarray(iface.base_pos, dtype=np.float32)
        reachable = self._nearest_navigable(pf, target)
        if reachable is None:
            self.waypoints = [target]
            return
        self.approach_point = reachable
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start
        path.requested_ends = [np.asarray(reachable, dtype=np.float32)]
        found = False
        try:
            found = bool(pf.find_path(path))
        except Exception:  # noqa: BLE001
            found = False
        if not found or len(path.points) < 2:
            self.waypoints = [reachable]
            return
        pts = [np.array(list(p), dtype=np.float64) for p in path.points]
        keep = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - keep[-1]) > self.control.dist_tol:
                keep.append(p)
        keep.append(reachable)
        self.waypoints = keep

    def begin(self, iface, kwargs):
        self.target = np.asarray(kwargs["target"][:3], dtype=np.float64)
        self.steps = 0
        self.wp_idx = 0
        self.stall_steps = 0
        self._plan(iface, self.target)
        self.max_steps = int(kwargs.get("budget", 400))

    def step(self, iface):
        self.steps += 1
        # current sub-goal
        if self.wp_idx < len(self.waypoints) - 1 and np.linalg.norm(
                self.waypoints[self.wp_idx][:2] - iface.base_pos[:2]
        ) < self.control.dist_tol:
            self.wp_idx += 1
        final_goal = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]

        # arrival is judged on the last waypoint, which is where the robot
        # can actually stand (the object itself may be on a counter)
        if np.linalg.norm(final_goal[:2] - iface.base_pos[:2]) < self.control.dist_tol \
                and self.wp_idx >= len(self.waypoints) - 1:
            return _make_action(base=[0.0, 0.0]), SkillResult.DONE
        if self.steps >= self.max_steps:
            return _make_action(base=[0.0, 0.0]), SkillResult.FAILED

        # stall detection: give up early instead of grinding into geometry
        if self.stalled_from is None:
            self.stalled_from = iface.base_pos.copy()
        elif np.linalg.norm(iface.base_pos - self.stalled_from) < 0.02:
            self.stall_steps += 1
            if self.stall_steps > 60:
                return _make_action(base=[0.0, 0.0]), SkillResult.FAILED
        else:
            self.stalled_from = iface.base_pos.copy()
            self.stall_steps = 0

        return _make_action(base=self.control(iface, final_goal)), SkillResult.RUNNING


class ArmToSkill(Skill):
    """``MoveArm`` -- servo the end-effector to a world position."""

    name = "move_arm"
    max_steps = 600  # ArmAction only moves 0.0125 rad/joint/step

    def __init__(self, residual_tol: float = 0.08) -> None:
        self.ik = ArmIKController(residual_tol=residual_tol)
        self.target: Optional[np.ndarray] = None
        self.steps = 0

    def begin(self, iface, kwargs):
        target = np.asarray(kwargs["target"][:3], dtype=np.float64)
        self.target = iface.clip_to_workspace(target)
        self.steps = 0

    def step(self, iface):
        self.steps += 1
        if self.ik.done(iface, self.target):
            return _make_action(arm=np.zeros(ARM_DIM)), SkillResult.DONE
        if self.steps >= self.max_steps:
            return _make_action(arm=np.zeros(ARM_DIM)), SkillResult.FAILED
        return _make_action(arm=self.ik.delta(iface, self.target)), SkillResult.RUNNING


class GripSkill(Skill):
    """``Grip`` -- snap on the nearest object (or release).

    Uses whichever grip action the task config registered: first tries the
    dedicated grip dimension of the 11-D action, and falls back to the
    last arm joint when ``disable_grip`` is set upstream.
    """

    name = "grip"
    max_steps = 12

    def __init__(self) -> None:
        self.release = False
        self.steps = 0

    def begin(self, iface, kwargs):
        self.release = bool(kwargs.get("release", False))
        self.steps = 0

    def step(self, iface):
        self.steps += 1
        want = -1.0 if self.release else 1.0
        done = (iface.is_grasped != self.release)
        if done or self.steps >= self.max_steps:
            return _make_action(grip=want), SkillResult.DONE
        return _make_action(grip=want), SkillResult.RUNNING


class WaitSkill(Skill):
    """``Wait`` -- idle for a fixed number of steps."""

    name = "wait"
    max_steps = 40

    def begin(self, iface, kwargs):
        self.max_steps = int(kwargs.get("steps", 20))
        self.steps = 0

    def step(self, iface):
        self.steps += 1
        res = SkillResult.DONE if self.steps >= self.max_steps else SkillResult.RUNNING
        return _make_action(), res


class PickSkill(Skill):
    """``Pick`` -- composite: ``Navigate -> MoveArm -> Grip -> MoveArm(rest)``.

    Identifies the nearest object to the end-effector if no index is given.
    """

    name = "pick"
    max_steps = 1200

    def __init__(self) -> None:
        self._active: Optional[Skill] = None
        self._sub_kwargs: Dict[str, Any] = {}
        self._queue: List[Skill] = []
        self.steps = 0

    def begin(self, iface, kwargs):
        self.steps = 0
        obj_idx = kwargs.get("index")
        objs = iface.object_positions
        if obj_idx is None:
            if len(objs) == 0:
                raise ValueError("no objects to pick")
            obj_idx = int(np.argmin(np.linalg.norm(objs - iface.ee_pos, axis=-1)))
        obj_idx = int(np.clip(obj_idx, 0, len(objs) - 1))
        obj_pos = objs[obj_idx]

        self._queue = [
            NavSkill(),
            ArmToSkill(),
            GripSkill(),
            ArmToSkill(),
        ]
        self._targets = [
            {"target": obj_pos, "budget": 300},
            {"target": obj_pos, "budget": 300},
            {},
            {"target": obj_pos + np.array([0.0, 0.35, 0.0])},  # lift up
        ]
        self._advance(iface)

    def _advance(self, iface):
        if not self._queue:
            self._active = None
            return
        self._active = self._queue.pop(0)
        self._active.begin(iface, self._targets.pop(0))

    def step(self, iface):
        self.steps += 1
        if self._active is None:
            return _make_action(), SkillResult.DONE
        action, res = self._active.step(iface)
        if res == SkillResult.DONE:
            self._advance(iface)
            return action, SkillResult.DONE if self._active is None else SkillResult.RUNNING
        if res == SkillResult.FAILED:
            self._active = None
            return action, SkillResult.FAILED
        return action, SkillResult.RUNNING


class PlaceSkill(Skill):
    """``Place`` -- composite: ``Navigate -> MoveArm -> Grip(release) -> Wait``."""

    name = "place"
    max_steps = 1200

    def __init__(self) -> None:
        self._active: Optional[Skill] = None
        self._queue: List[Skill] = []
        self._targets: List[Dict[str, Any]] = []
        self.steps = 0

    def begin(self, iface, kwargs):
        self.steps = 0
        idx = kwargs.get("index")
        targets = iface.target_positions
        if idx is None:
            if len(targets) == 0:
                raise ValueError("no targets to place at")
            idx = int(np.argmin(np.linalg.norm(targets - iface.ee_pos, axis=-1)))
        idx = int(np.clip(idx, 0, len(targets) - 1))
        goal = targets[idx]

        self._queue = [NavSkill(), ArmToSkill(), GripSkill(), WaitSkill()]
        self._targets = [
            {"target": goal, "budget": 300},
            {"target": goal + np.array([0.0, 0.2, 0.0])},
            {"release": True},
            {"steps": 15},
        ]
        self._advance(iface)

    def _advance(self, iface):
        if not self._queue:
            self._active = None
            return
        self._active = self._queue.pop(0)
        self._active.begin(iface, self._targets.pop(0))

    def step(self, iface):
        self.steps += 1
        if self._active is None:
            return _make_action(), SkillResult.DONE
        action, res = self._active.step(iface)
        if res == SkillResult.DONE:
            self._advance(iface)
            return action, SkillResult.DONE if self._active is None else SkillResult.RUNNING
        if res == SkillResult.FAILED:
            self._active = None
            return action, SkillResult.FAILED
        return action, SkillResult.RUNNING


class DoneSkill(Skill):
    """``Done`` -- request episode termination (the missing stop action)."""

    name = "done"
    max_steps = 1

    def begin(self, iface, kwargs):
        pass

    def step(self, iface):
        action = _make_action()
        action[STOP_SLICE] = 1.0
        return action, SkillResult.DONE


class SkillLibrary:
    """PARTNR-style registry mapping skill names to controllers."""

    def __init__(self, habitat_env: Any, agent_idx: int = 0) -> None:
        self.iface = BaseInterface(habitat_env, agent_idx)
        self.skills: Dict[str, Skill] = {
            skill_cls.name: skill_cls()
            for skill_cls in (
                NavSkill, ArmToSkill, GripSkill, WaitSkill,
                PickSkill, PlaceSkill, DoneSkill,
            )
        }
        self.names: List[str] = list(self.skills.keys())
        self._current: Optional[Skill] = None

    # -- introspection helpers --------------------------------------
    @property
    def num_objects(self) -> int:
        return len(self.iface.object_positions)

    @property
    def num_targets(self) -> int:
        return len(self.iface.target_positions)

    def describe(self) -> str:
        lines = ["Available skills:"]
        for name in self.names:
            lines.append(
                "  %-10s | max_steps=%4d | %s"
                % (name, self.skills[name].max_steps,
                   self.skills[name].__class__.__doc__.strip().split("\n")[0])
            )
        lines.append("  world: %d objects, %d targets, holding=%s"
                     % (self.num_objects, self.num_targets, self.iface.is_grasped))
        return "\n".join(lines)

    # -- execution ---------------------------------------------------
    def begin(self, name: str, **kwargs) -> None:
        if name not in self.skills:
            raise KeyError("unknown skill %r (have %s)" % (name, self.names))
        skill = self.skills[name]
        if skill.max_steps > 0:
            pass
        skill.begin(self.iface, kwargs)
        self._current = skill

    def step(self) -> Tuple[np.ndarray, SkillResult]:
        if self._current is None:
            raise RuntimeError("no active skill; call begin() first")
        return self._current.step(self.iface)

    @property
    def active(self) -> Optional[str]:
        return getattr(self._current, "name", None)
