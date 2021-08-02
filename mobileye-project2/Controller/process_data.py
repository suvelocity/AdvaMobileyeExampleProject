
import numpy as np

from Model.files_utils import read_txt, read_pickle
from Model.frame_container import FrameContainer


class ProcessData:
    def __init__(self, pls_file):
        self.pls_file = pls_file

    def process_data(self, frame):
        lines = read_txt(self.pls_file)
        pkl_path = lines[0]
        data = read_pickle(pkl_path)

        frame.init_focal_pp(data['flx'], data['principle_point'])

        for img_path in lines[1:]:
            curr_container = FrameContainer(img_path)
            curr_container.id = int(img_path[42:44])  # Extract the frame number from the line
            curr_container.traffic_light = np.array(data['points_' + str(curr_container.id)][0])
            EM = np.eye(4)

            # for i in range(curr_container.id - 1, curr_container.id):
            #     EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            curr_container.EM = np.dot(data['egomotion_' + str(curr_container.id - 1) + '-' + str(curr_container.id)], EM)
            frame.update_frame(curr_container)
            yield curr_container.id

    def count_file_lines(self):
        lines = read_txt(self.pls_file)
        return len(lines) - 1

    def first_id(self):
        lines = read_txt(self.pls_file)
        return int(lines[1][31:33])
