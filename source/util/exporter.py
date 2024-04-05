import os


class Exporter:
    results = os.path.join('results')
    quadeds = os.path.join('results', 'quadrangulated')
    ngftchs = os.path.join('results', 'torched')
    ngfbins = os.path.join('results', 'binaries')

    @staticmethod
    def dirfill():
        os.makedirs(Exporter.results, exist_ok=True)
        os.makedirs(Exporter.quadeds, exist_ok=True)
        os.makedirs(Exporter.ngftchs, exist_ok=True)
        os.makedirs(Exporter.ngfbins, exist_ok=True)

    def __init__(self, mesh: str, lod: int):
        Exporter.dirfill()

        self.prefix = os.path.basename(mesh)
        self.prefix = self.prefix.split('.')[0]
        self.lod = lod

    def quaded(self):
        return os.path.join(Exporter.quadeds, self.prefix + f'-lod{self.lod}.obj')

    def torched(self):
        return os.path.join(Exporter.ngftchs, self.prefix + f'-lod{self.lod}.pt')

    def binary(self):
        return os.path.join(Exporter.ngfbins, self.prefix + f'-lod{self.lod}.bin')
