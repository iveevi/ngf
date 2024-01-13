import os
import re
import torch
import prettytable

from ngf import load_ngf

if __name__ == '__main__':
    # Use XYZ model as the benchmark
    lods = {}

    pattern = re.compile('^lod\d.pt$')
    for root, _, files in os.walk('results/xyz'):
        for file in files:
            if pattern.match(file):
                file = os.path.join(root, file)

                ngf = torch.load(file)
                ngf = load_ngf(ngf)

                count = ngf.complexes.shape[0]
                count = 10 * ((count + 9) // 10)
                lods[count] = ngf

    print('LODS', lods.keys())

    # Timings for gradient and non-gradient pass
    # timings = { 'grad', 'no_grad' }
    timings = {}

    for lod in lods:
        ngf = lods[lod]

        # With gradients
        for tessellation in [ 4, 8, 16 ]:
            print('Profiling lod %d with gradient pass at resolution %d' % (lod, tessellation))
            with torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=10)) as prof:
                for i in range(100):
                    with torch.profiler.record_function('evaluation'):
                        with torch.no_grad():
                            uvs = ngf.sample_uniform(tessellation)
                        V = ngf.eval(*uvs).detach()
                        prof.step()

            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            data = None
            for entry in prof.key_averages():
                if entry.key == 'evaluation':
                    data = entry
                    break

            cuda_time = data.cuda_time_total / (data.count * 1e3)
            # print('  > time (CPU %.2f ms, %.2f CUDA ms)' %(data.cpu_time_total / (data.count * 1000), data.cuda_time_total / (data.count * 1000)))

            timings.setdefault(lod, {})[tessellation] = cuda_time

        # Without gradients
        # print('Profiling lod %d with gradient pass' % lod)
        #
        # ngf.mlp.eval()
        # with torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=10)) as prof:
        #     for i in range(10):
        #         with torch.profiler.record_function('evaluation'):
        #             with torch.no_grad():
        #                 uvs = ngf.sample_uniform(16)
        #                 V = ngf.eval(*uvs).detach()
        #             prof.step()
        # data = None
        # for entry in prof.key_averages():
        #     if entry.key == 'evaluation':
        #         data = entry
        #         break
        #
        # print('  > time (CPU %.2f ms, %.2f CUDA ms)' %(data.cpu_time_total / (data.count * 1000), data.cuda_time_total / (data.count * 1000)))

    print('timings', timings)

    table = prettytable.PrettyTable()
    table.field_names = [ 'Patches', 4, 8, 16 ]
    for lod in timings:
        row = [ lod, '%.2f ms' % timings[lod][4], '%.2f ms' % timings[lod][8], '%.2f ms' % timings[lod][16] ]
        table.add_row(row)

    print(table.get_latex_string())
