"""Dependency-free top-down RGB renderer for headless servers."""

from __future__ import annotations

import math

import numpy as np

from .types import Team


class RGBRenderer:
    def __init__(self, width=960, height=640, margin=55):
        self.width = width
        self.height = height
        self.margin = margin

    def render(self, state, config):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:] = (25, 112, 55)

        def point(x, y):
            px = self.margin + (x + config.field_length / 2.0) / config.field_length * (self.width - 2 * self.margin)
            py = self.height - self.margin - (y + config.field_width / 2.0) / config.field_width * (self.height - 2 * self.margin)
            return int(round(px)), int(round(py))

        white = (235, 235, 235)
        x0, y0 = point(-config.field_length / 2.0, config.field_width / 2.0)
        x1, y1 = point(config.field_length / 2.0, -config.field_width / 2.0)
        self._rect(image, x0, y0, x1, y1, white, 3)
        cx0, cy0 = point(0.0, config.field_width / 2.0)
        cx1, cy1 = point(0.0, -config.field_width / 2.0)
        self._line(image, cx0, cy0, cx1, cy1, white, 2)
        center = point(0.0, 0.0)
        radius_px = int(config.center_circle_radius / config.field_length * (self.width - 2 * self.margin))
        self._circle_outline(image, center[0], center[1], radius_px, white, 2)

        for sign in (-1.0, 1.0):
            goal_x = sign * config.field_length / 2.0
            inner_x = goal_x - sign * config.penalty_area_length
            a = point(goal_x, config.penalty_area_width / 2.0)
            b = point(inner_x, -config.penalty_area_width / 2.0)
            self._rect(image, a[0], a[1], b[0], b[1], white, 2)
            goal_outer = goal_x + sign * config.goal_depth
            a = point(goal_x, config.goal_width / 2.0)
            b = point(goal_outer, -config.goal_width / 2.0)
            self._rect(image, a[0], a[1], b[0], b[1], (205, 205, 205), 2)

        scale_x = (self.width - 2 * self.margin) / config.field_length
        robot_radius = max(5, int(config.robot_radius * scale_x))
        for robot in state.robots.values():
            x, y = point(robot.pose.x, robot.pose.y)
            color = (45, 115, 255) if robot.team is Team.BLUE else (235, 65, 65)
            if not robot.active:
                color = (110, 110, 110)
            self._disk(image, x, y, robot_radius, color)
            hx = x + int(math.cos(robot.pose.theta) * robot_radius * 1.5)
            hy = y - int(math.sin(robot.pose.theta) * robot_radius * 1.5)
            self._line(image, x, y, hx, hy, (20, 20, 20), 2)

        bx, by = point(state.ball.x, state.ball.y)
        self._disk(image, bx, by, max(3, int(config.ball_radius * scale_x)), (245, 245, 245))
        return image

    @staticmethod
    def _disk(image, cx, cy, radius, color):
        y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        image[y0:y1, x0:x1][mask] = color

    @classmethod
    def _circle_outline(cls, image, cx, cy, radius, color, thickness):
        cls._disk(image, cx, cy, radius, color)
        if radius > thickness:
            cls._disk(image, cx, cy, radius - thickness, (25, 112, 55))

    @staticmethod
    def _line(image, x0, y0, x1, y1, color, thickness=1):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1).astype(int)
        ys = np.linspace(y0, y1, steps + 1).astype(int)
        for offset_x in range(-thickness // 2, thickness // 2 + 1):
            for offset_y in range(-thickness // 2, thickness // 2 + 1):
                xx = np.clip(xs + offset_x, 0, image.shape[1] - 1)
                yy = np.clip(ys + offset_y, 0, image.shape[0] - 1)
                image[yy, xx] = color

    @classmethod
    def _rect(cls, image, x0, y0, x1, y1, color, thickness=1):
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        cls._line(image, left, top, right, top, color, thickness)
        cls._line(image, right, top, right, bottom, color, thickness)
        cls._line(image, right, bottom, left, bottom, color, thickness)
        cls._line(image, left, bottom, left, top, color, thickness)

