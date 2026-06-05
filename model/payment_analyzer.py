from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]
NormalizedBox = Tuple[float, float, float, float]


class PassengerStatus(str, Enum):
    PAID = "paid"
    SUSPECT = "suspect"


@dataclass
class PersonTrack:
    track_id: int
    box: Box
    center: Point
    wrist_points: Tuple[Point, ...] = ()


@dataclass
class TrackMemory:
    stable_id: int
    box: Box
    center: Point
    last_seen_frame: int
    yolo_ids: set = field(default_factory=set)


@dataclass
class PaymentEvent:
    track_id: int
    status: PassengerStatus
    message: str
    frame_index: int


@dataclass
class PassengerState:
    track_id: int
    status: PassengerStatus = PassengerStatus.SUSPECT
    frames_seen: int = 0
    frames_with_hand_near_terminal: int = 0
    frames_in_cabin_without_payment: int = 0
    last_center: Point = (0, 0)
    last_box: Box = (0, 0, 0, 0)
    last_hand_box: Optional[Box] = None
    last_wrist_points: Tuple[Point, ...] = ()
    emitted_statuses: set = field(default_factory=set)


class PaymentAnalyzer:
    """
    Анализатор NFC-платежей, построенный на основе отслеживания местоположения человека.
    Платёж подтверждается положением запястья относительно терминала
    """

    def __init__(
        self,
        terminal_zone: NormalizedBox = (0.02, 0.18, 0.34, 0.78),
        hand_contact_frames_threshold: int = 3,
        suspect_frames_in_cabin: int = 25,
        stale_after_frames: int = 90,
    ):
        self.terminal_zone = terminal_zone
        self.hand_contact_frames_threshold = hand_contact_frames_threshold
        self.suspect_frames_in_cabin = suspect_frames_in_cabin
        self.stale_after_frames = stale_after_frames
        self.frame_index = 0
        self.next_stable_id = 1
        self.track_memory: Dict[int, TrackMemory] = {}
        self.yolo_to_stable: Dict[int, int] = {}
        self.states: Dict[int, PassengerState] = {}

    def reset(self):
        self.frame_index = 0
        self.next_stable_id = 1
        self.track_memory.clear()
        self.yolo_to_stable.clear()
        self.states.clear()

    def analyze(self, result) -> List[PaymentEvent]:
        self.frame_index += 1
        self.assigned_stable_ids = set()
        events: List[PaymentEvent] = []
        tracks = list(self._extract_tracks(result))
        active_ids = {track.track_id for track in tracks}

        for track in tracks:
            state = self.states.setdefault(
                track.track_id, PassengerState(track_id=track.track_id)
            )
            state.frames_seen += 1
            state.last_center = track.center
            state.last_box = track.box
            state.last_wrist_points = track.wrist_points
            state.last_hand_box = (
                None if track.wrist_points else self._estimate_hand_box(track.box, result.orig_shape)
            )

            if self._is_hand_near_terminal(track, state.last_hand_box, result.orig_shape):
                state.frames_with_hand_near_terminal += 1
                state.frames_in_cabin_without_payment = 0
            else:
                state.frames_with_hand_near_terminal = 0
                if self._is_in_cabin_area(track.center, result.orig_shape):
                    state.frames_in_cabin_without_payment += 1

            if state.frames_with_hand_near_terminal >= self.hand_contact_frames_threshold:
                state.status = PassengerStatus.PAID
                events.append(
                    self._event_once(
                        state,
                        PassengerStatus.PAID,
                        f"Passenger #{track.track_id} hand detected near terminal",
                    )
                )

        self._drop_stale_tracks(active_ids)
        return [event for event in events if event is not None]

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        zone = self._scale_box(self.terminal_zone, (h, w))
        contact_zone = self._expand_box(zone, frame.shape, 0.2)

        cv2.rectangle(frame, zone[:2], zone[2:], (0, 180, 255), 2)
        cv2.rectangle(frame, contact_zone[:2], contact_zone[2:], (0, 120, 255), 1)
        cv2.putText(
            frame,
            "TERMINAL",
            (zone[0], max(20, zone[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

        for state in self.states.values():
            color = self._status_color(state.status)
            x1, y1, x2, y2 = state.last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {state.track_id}: {self._status_label(state.status)}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            for wrist in state.last_wrist_points:
                cv2.circle(frame, wrist, 6, (80, 255, 255), -1)
                cv2.circle(frame, wrist, 9, (0, 0, 0), 1)

            if state.last_hand_box:
                hx1, hy1, hx2, hy2 = state.last_hand_box
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (80, 255, 255), 1)
                cv2.putText(
                    frame,
                    "HAND AREA",
                    (hx1, max(20, hy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (80, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return frame

    def _extract_tracks(self, result) -> Iterable[PersonTrack]:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        ids = boxes.id.cpu().int().tolist()
        xyxy = boxes.xyxy.cpu().int().tolist()
        wrist_points_by_index = self._extract_wrist_points(result)
        tracks = []

        for index, (track_id, box) in enumerate(zip(ids, xyxy)):
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            stable_id = self._resolve_stable_id(
                int(track_id),
                (x1, y1, x2, y2),
                center,
                result.orig_shape,
            )
            tracks.append(
                PersonTrack(
                    track_id=stable_id,
                    box=(x1, y1, x2, y2),
                    center=center,
                    wrist_points=wrist_points_by_index.get(index, ()),
                )
            )
        return tracks

    def _resolve_stable_id(
        self,
        yolo_id: int,
        box: Box,
        center: Point,
        frame_shape,
    ) -> int:
        mapped_id = self.yolo_to_stable.get(yolo_id)
        if mapped_id in self.track_memory and mapped_id not in self.assigned_stable_ids:
            self._update_track_memory(mapped_id, yolo_id, box, center)
            return mapped_id

        match_id = self._match_existing_track(box, center, frame_shape)
        if match_id is None:
            match_id = self.next_stable_id
            self.next_stable_id += 1

        self.yolo_to_stable[yolo_id] = match_id
        self._update_track_memory(match_id, yolo_id, box, center)
        return match_id

    def _match_existing_track(self, box: Box, center: Point, frame_shape) -> Optional[int]:
        h, w = frame_shape[:2]
        diagonal = max(1.0, (w * w + h * h) ** 0.5)
        best_id = None
        best_score = -1.0

        for stable_id, memory in self.track_memory.items():
            if stable_id in self.assigned_stable_ids:
                continue

            age = self.frame_index - memory.last_seen_frame
            if age > self.stale_after_frames:
                continue

            iou = self._box_iou(box, memory.box)
            distance = self._point_distance(center, memory.center) / diagonal
            score = iou - distance - age * 0.002

            if (iou >= 0.12 or distance <= 0.10) and score > best_score:
                best_score = score
                best_id = stable_id

        return best_id

    def _update_track_memory(self, stable_id: int, yolo_id: int, box: Box, center: Point):
        memory = self.track_memory.get(stable_id)
        if memory is None:
            memory = TrackMemory(
                stable_id=stable_id,
                box=box,
                center=center,
                last_seen_frame=self.frame_index,
            )
            self.track_memory[stable_id] = memory

        memory.box = box
        memory.center = center
        memory.last_seen_frame = self.frame_index
        memory.yolo_ids.add(yolo_id)
        self.assigned_stable_ids.add(stable_id)

    def _extract_wrist_points(self, result) -> Dict[int, Tuple[Point, ...]]:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None:
            return {}

        xy = keypoints.xy.cpu().tolist()
        conf = None
        if keypoints.conf is not None:
            conf = keypoints.conf.cpu().tolist()

        wrist_points_by_index: Dict[int, Tuple[Point, ...]] = {}
        for index, person_keypoints in enumerate(xy):
            points = []
            for wrist_index in (9, 10):
                if wrist_index >= len(person_keypoints):
                    continue
                if conf is not None and conf[index][wrist_index] < 0.35:
                    continue

                x, y = person_keypoints[wrist_index]
                if x <= 0 and y <= 0:
                    continue
                points.append((int(x), int(y)))

            if points:
                wrist_points_by_index[index] = tuple(points)

        return wrist_points_by_index

    def _is_hand_near_terminal(
        self,
        track: PersonTrack,
        fallback_hand_box: Optional[Box],
        frame_shape,
    ) -> bool:
        terminal_box = self._scale_box(self.terminal_zone, frame_shape[:2])
        contact_zone = self._expand_box(terminal_box, frame_shape, 0.2)

        if track.wrist_points:
            return any(
                self._point_in_box(wrist, contact_zone)
                for wrist in track.wrist_points
            )

        if fallback_hand_box is None:
            return False
        return self._boxes_intersect(fallback_hand_box, contact_zone)

    def _estimate_hand_box(self, person_box: Box, frame_shape) -> Box:
        h, w = frame_shape[:2]
        terminal_box = self._scale_box(self.terminal_zone, (h, w))
        terminal_center_x = (terminal_box[0] + terminal_box[2]) // 2

        x1, y1, x2, y2 = person_box
        body_w = max(1, x2 - x1)
        body_h = max(1, y2 - y1)
        person_center_x = (x1 + x2) // 2

        if terminal_center_x < person_center_x:
            hand_x1 = x1 - int(body_w * 0.55)
            hand_x2 = x1 + int(body_w * 0.25)
        else:
            hand_x1 = x2 - int(body_w * 0.25)
            hand_x2 = x2 + int(body_w * 0.55)

        hand_y1 = y1 + int(body_h * 0.22)
        hand_y2 = y1 + int(body_h * 0.68)

        return (
            max(0, hand_x1),
            max(0, hand_y1),
            min(w - 1, hand_x2),
            min(h - 1, hand_y2),
        )

    def _event_once(
        self,
        state: PassengerState,
        status: PassengerStatus,
        message: str,
    ) -> Optional[PaymentEvent]:
        if status in state.emitted_statuses:
            return None

        state.emitted_statuses.add(status)
        return PaymentEvent(
            track_id=state.track_id,
            status=status,
            message=message,
            frame_index=self.frame_index,
        )

    def _drop_stale_tracks(self, active_ids: set):
        stale_ids = []
        for track_id, state in self.states.items():
            if track_id in active_ids:
                continue

            state.frames_in_cabin_without_payment += 1
            if state.frames_in_cabin_without_payment >= self.stale_after_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.states[track_id]

        stale_memory_ids = [
            stable_id
            for stable_id, memory in self.track_memory.items()
            if self.frame_index - memory.last_seen_frame >= self.stale_after_frames
        ]
        for stable_id in stale_memory_ids:
            memory = self.track_memory.pop(stable_id)
            for yolo_id in memory.yolo_ids:
                if self.yolo_to_stable.get(yolo_id) == stable_id:
                    del self.yolo_to_stable[yolo_id]

    def _is_in_cabin_area(self, point: Point, frame_shape) -> bool:
        h, w = frame_shape[:2]
        x1, _, x2, _ = self._scale_box(self.terminal_zone, (h, w))
        terminal_mid = (x1 + x2) // 2
        x, y = point
        moved_past_terminal = x > terminal_mid if terminal_mid < w // 2 else x < terminal_mid
        return moved_past_terminal and int(0.1 * h) <= y <= int(0.95 * h)

    @staticmethod
    def _scale_box(box: NormalizedBox, frame_size: Tuple[int, int]) -> Box:
        h, w = frame_size
        x1, y1, x2, y2 = box
        return (
            int(min(x1, x2) * w),
            int(min(y1, y2) * h),
            int(max(x1, x2) * w),
            int(max(y1, y2) * h),
        )

    @staticmethod
    def _expand_box(box: Box, frame_shape, ratio: float) -> Box:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * ratio)
        pad_y = int((y2 - y1) * ratio)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w - 1, x2 + pad_x),
            min(h - 1, y2 + pad_y),
        )

    @staticmethod
    def _boxes_intersect(a: Box, b: Box) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1

    @staticmethod
    def _box_iou(a: Box, b: Box) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return intersection / max(1, area_a + area_b - intersection)

    @staticmethod
    def _point_distance(a: Point, b: Point) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _point_in_box(point: Point, box: Box) -> bool:
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _status_color(status: PassengerStatus):
        if status == PassengerStatus.PAID:
            return (70, 220, 70)
        if status == PassengerStatus.SUSPECT:
            return (40, 40, 240)
        return (255, 220, 80)

    @staticmethod
    def _status_label(status: PassengerStatus) -> str:
        if status == PassengerStatus.PAID:
            return "PAID"
        return "NO PAY"
