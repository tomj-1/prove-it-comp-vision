import cv2

from LaneDetection import LaneDetection
from YoloPV2ExternalStuff.utils.utils import plot_one_box, scale_coords
from ObjectDetection import ObjectDetection
class Visualizer:

    def __init__(self, img, model,  opt):
        self.img = img
        self.model = model
        self.opt = opt


    def visualize(self):

        lane_mask = LaneDetection(
            self.model,
            self.img
        ).detectlane()

        detections = ObjectDetection(
            self.model,
            self.opt,
            self.img
        ).detectobject()

        # Resize lane mask to match original image
        lane_mask_resized = cv2.resize(
            lane_mask,
            (self.img.shape[1], self.img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Color detected lane pixels yellow
        self.img[lane_mask_resized > 0] = (0, 255, 255)

        # Draw object boxes
        if len(detections):

            detections[:, :4] = scale_coords(
                (640, 640),
                detections[:, :4],
                self.img.shape
            ).round()

            for *xyxy, conf, cls in reversed(detections):

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

        previous_element = None
        
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape
        
        x_scale = img_w / mask_w
        y_scale = img_h / mask_h
        

        for element in list_of_points:
            
            element = (
                int(element[0] * x_scale),
                int(element[1] * y_scale)
            )

            if previous_element is not None:

                if (
                    abs(previous_element[0] - element[0] )
                    < img_w // 5
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