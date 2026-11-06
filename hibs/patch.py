import scene.scene_model
import train

from hibs import storage

_installed = False


def install():
    global _installed
    if _installed:
        return
    from hibs.dataset import HibsImageDataset
    from hibs.keyframe import HibsKeyframe
    from hibs.registration import GatedPoseInitializer
    from hibs.scene import HibsSceneModel
    from hibs.viewer import HibsViewer

    scene.scene_model.HierarchyStructure = storage.BetaHierarchy
    scene.scene_model.Keyframe = HibsKeyframe
    train.SceneModel = HibsSceneModel
    train.PoseInitializer = GatedPoseInitializer
    train.GaussianViewer = HibsViewer
    train.Keyframe = HibsKeyframe
    train.ImageDataset = HibsImageDataset
    _installed = True
