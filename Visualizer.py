import cv2

from LaneDetection import LaneDetection
from import plot_one_box


class Visualizer:

    def __init__(self, img, model, calibration_points, detections):
        self.img = img
        self.model = model
        self.calibration_points = calibration_points
        self.detections = detections


    def visualize(self):

        # Get lane mask
        lane_mask = LaneDetection(
            self.model,
            self.img
        ).detectlane()

        left_lane_points = []
        right_lane_points = []
        left_left_lane_points = []
        right_right_lane_points = []

        bottom_horizon = self.calibration_points[4]
        upper_horizon = self.calibration_points[5]

        D = bottom_horizon[1] - upper_horizon[1]

        img_middle = self.img.shape[1] // 2

        for i in range(20):

            horizontal_line = bottom_horizon[1] - (i * D // 20)

            points = find_middle_pixel_on_height(
                lane_mask,
                horizontal_line
            )

            (
                left_left_lane_points,
                left_lane_points,
                right_lane_points,
                right_right_lane_points
            ) = separate_points(
                points,
                left_left_lane_points,
                left_lane_points,
                right_lane_points,
                right_right_lane_points,
                img_middle
            )

        # Draw left lane
        self.img = display_from_list(
            self.img,
            left_lane_points,
            lane_mask,
            (0, 255, 255)
        )

        # Draw right lane
        self.img = display_from_list(
            self.img,
            right_lane_points,
            lane_mask,
            (0, 255, 255)
        )



        if len(self.detections):

            for *xyxy, conf, cls in reversed(self.detections):

                plot_one_box(
                    xyxy,
                    self.img,
                    color=[0, 0, 255],
                    line_thickness=2
                )


        return self.img


def find_middle_pixel_on_height(lane_mask, height):

    horizontal_lane = lane_mask[height]

    cnt1 = 0
    previous_pixel = 0
    points_list = []

    for current_pixel in range(len(horizontal_lane)):

        if (
            horizontal_lane[previous_pixel]
            * horizontal_lane[current_pixel]
            == 1
        ):
            cnt1 += 1

        elif (
            horizontal_lane[previous_pixel]
            * horizontal_lane[current_pixel]
            == 0
            and cnt1 != 0
        ):

            points_list.append(
                (
                    current_pixel - cnt1 // 2,
                    height
                )
            )

            cnt1 = 0

        previous_pixel = current_pixel

    return points_list


def separate_points(
    points,
    left_left_lane_points,
    left_lane_points,
    right_lane_points,
    right_right_lane_points,
    img_middle
):

    left_distance_list = []
    right_distance_list = []

    for point in points:

        distance = point[0] - img_middle

        if distance < 0:
            left_distance_list.append(abs(distance))

        else:
            right_distance_list.append(abs(distance))

    if len(left_distance_list) > 0:

        min_left_idx = left_distance_list.index(
            min(left_distance_list)
        )

        left_lane_points.append(
            points[min_left_idx]
        )

        if len(left_distance_list) > 1:
            left_left_lane_points.append(
                points[min_left_idx - 1]
            )

    if len(right_distance_list) > 0:

        min_right_idx = right_distance_list.index(
            min(right_distance_list)
        )

        min_right_idx += len(left_distance_list)

        right_lane_points.append(
            points[min_right_idx]
        )

        if len(right_distance_list) > 1:

            right_right_lane_points.append(
                points[min_right_idx + 1]
            )

    return (
        left_left_lane_points,
        left_lane_points,
        right_lane_points,
        right_right_lane_points
    )


def display_from_list(
    img,
    list_of_points,
    mask,
    color
):

    previous_element = []

    for element in list_of_points:

        if len(previous_element) != 0:

            if (
                abs(previous_element[0] - element[0])
                < mask.shape[1] // 5
            ):

                cv2.line(
                    img,
                    previous_element,
                    element,
                    color,
                    2
                )

        previous_element = element

    return img