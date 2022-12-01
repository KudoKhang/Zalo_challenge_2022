import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IECore


class OpenVinoModel:
    def __init__(self, model_path, input_size=(320, 320), device_name="CPU"):
        self.input_size = input_size
        # Load model
        ie = IECore()
        model = model_path
        log.info(f"Loading network:\n\t{model}")
        net = ie.read_network(model=model)

        # Defind in, out
        self.input_blob = next(iter(net.input_info))
        out_blob = next(iter(net.outputs))

        # Get input
        n, c, h, w = net.input_info[self.input_blob].input_data.shape
        print("Input shape", n, c, h, w)

        # Load model to plugin
        print("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name=device_name)
        self.threshold = 0.2
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2
        self.nms_threshold = 0.4

    def predict(self, image):
        image_shape = image.shape
        image = self._preprocess(image)
        res = self.exec_net.infer(inputs={self.input_blob: image})
        res = list(res.values())
        bboxes = self._postprocess(res, image_shape)
        return bboxes

    def _preprocess(self, image):
        # To Square image
        h, w, c = image.shape
        image_size = h if h >= w else w
        self.start_h = 0 if h > w else int((w - h) / 2)
        self.start_w = 0 if w > h else int((h - w) / 2)
        square_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        square_image[self.start_h:self.start_h + h, self.start_w:self.start_w + w] = image
        image = square_image
        image = np.expand_dims(cv2.resize(image, self.input_size), axis=0)
        image = image.transpose((0, 3, 1, 2))
        return image

    def _postprocess(self, output, image_shape):
        self.input_height = 320
        self.input_width = 320
        scores_list = []
        bboxes_list = []
        # debug
        for i in range(len(output)):
            output[i] = output[i].astype(np.float16)
        # process
        for index, stride in enumerate(self.feat_stride_fpn):
            scores = output[index * 2]
            bbox_preds = output[index * 2 + 1]
            bbox_preds = bbox_preds * stride
            height = self.input_height // stride
            width = self.input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= self.threshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        new_bboxes = []
        for bbox in det:
            new_bboxes.append(self.get_face_location(bbox, image_shape))

        return new_bboxes

    def nms(self, dets):
        thresh = self.nms_threshold
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def softmax(z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div

    @staticmethod
    def distance2bbox(points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def get_face_location(self, bbox, shape):
        h, w, c = shape
        h = w if h < w else h
        w = h if w < h else w
        scale = 0
        # get face location
        x_min, y_min, x_max, y_max, score = \
            (bbox * [w / self.input_width, h / self.input_height, w / self.input_width, h / self.input_height, 1]) \
                .astype(np.int64)

        x_min -= (x_max - x_min) * scale
        x_max += (x_max - x_min) * scale
        y_min -= (y_max - y_min) * scale
        y_max += (y_max - y_min) * scale

        if x_max - x_min > y_max - y_min:
            y_max += ((x_max - x_min) - (y_max - y_min)) / 2
            y_min -= ((x_max - x_min) - (y_max - y_min)) / 2
        else:
            x_max += ((y_max - y_min) - (x_max - x_min)) / 2
            x_min -= ((y_max - y_min) - (x_max - x_min)) / 2

        x_min -= self.start_w
        y_min -= self.start_h
        x_max -= self.start_w
        y_max -= self.start_h

        y_min = 0 if y_min < 0 else int(y_min)
        y_max = h if y_max > h else int(y_max)
        x_min = 0 if x_min < 0 else int(x_min)
        x_max = w if x_max > w else int(x_max)

        return x_min, y_min, x_max, y_max, score

