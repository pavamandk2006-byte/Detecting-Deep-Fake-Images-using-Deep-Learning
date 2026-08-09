import cv2
import numpy as np
import torch


class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )


    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output.detach()


    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0].detach()
        )


    def generate(
        self,
        input_tensor,
        class_index=None
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            input_tensor
        )

        if class_index is None:

            class_index = int(
                torch.argmax(
                    output,
                    dim=1
                ).item()
            )

        score = output[
            :,
            class_index
        ].sum()

        score.backward()

        if (
            self.activations is None
            or
            self.gradients is None
        ):

            raise RuntimeError(
                "Grad-CAM hooks did not receive data."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * self.activations
        ).sum(
            dim=1
        )

        cam = torch.relu(
            cam
        )

        cam = cam[0].cpu().numpy()

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 1e-8:

            cam /= maximum

        cam_uint8 = np.uint8(
            cam * 255
        )

        heatmap_bgr = cv2.applyColorMap(
            cam_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB
        )

        return (
            heatmap_rgb,
            class_index
        )


    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


def overlay_heatmap(
    original_bgr,
    heatmap_bgr,
    alpha=0.42
):

    height, width = (
        original_bgr.shape[:2]
    )

    heatmap_bgr = cv2.resize(
        heatmap_bgr,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return cv2.addWeighted(
        original_bgr,
        1.0 - alpha,
        heatmap_bgr,
        alpha,
        0
    )