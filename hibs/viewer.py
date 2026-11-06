from gaussianviewer import GaussianViewer

from hibs.confidence import confidence_field

CONFIDENCE_MODE = "Beta Field"


class HibsViewer(GaussianViewer):
    def create_widgets(self):
        super().create_widgets()
        self.render_modes.insert(2, CONFIDENCE_MODE)

    def render_mode(self):
        mode = self.render_modes[self.render_mode_id]
        return "Splats" if mode == CONFIDENCE_MODE else mode

    def step(self):
        if self.render_modes[self.render_mode_id] != CONFIDENCE_MODE:
            return super().step()
        with confidence_field(self.scene_model):
            super().step()
