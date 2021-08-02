
from Model.data import Data
from .process_data import ProcessData
from .TFL_manager import TFLManager


class Controller:
    def process(self, pls_path):
        data = Data()
        p = ProcessData(pls_path)
        manager = TFLManager()

        for frame in p.process_data(data):
            manager.run(data, frame)
