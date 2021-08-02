
try:
    print("tensorflow/numpy imports:")
    import numpy as np
    from tensorflow.keras.models import load_model

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")

from Controller.adapter import Adapter
from Model.images_utils import crop_image
from Model.SFM import calc_TFL_dist
from Model.run_attention import find_tfl_lights
from View.output import Output


class TFLManager:
    def run(self, data, img_id):
        adapter = Adapter()
        image = np.array(data.curr.img)
        candidates, auxiliary = adapter.adapt_part_1_to_part_2(find_tfl_lights(image))
        cropped_imgs = crop_image(image, candidates)

        loaded_model = load_model("Model\data\model.h5")
        cropped_imgs_predicts = loaded_model.predict(np.array(cropped_imgs))

        # tmp = [(candidates[i], percent) for i, percent in enumerate(percents[:, 1]) if percent > 0.8]
        # tfl_candidates = sorted(tmp[:20], key=lambda k: k[1], reverse=True)[:10]
        # data.curr.traffic_light = np.array([x for x, y in tfl_candidates])
        candidates_list = adapter.adapt_part_2_to_part_3(cropped_imgs_predicts, candidates)
        data.curr.traffic_light = np.array(candidates_list)

        if data.prev and cropped_imgs_predicts.shape[0] <= len(cropped_imgs):
            data.curr = calc_TFL_dist(data.prev, data.curr, data.focal, data.pp)
            Output.visualize(data, img_id)

