from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt

def read_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    data = [line.replace('\n','') for line in data]
    def line_below(specific_phrase, end = 1):
        if isinstance(end, int):
            for i in range(len(data)):
                if data[i]==specific_phrase:
                    return data[i+1: i+1+end]
        else:
            section = []
            recording=False
            for line in data:
                if line==end: # stopping condition first
                    recording=False
                if recording:
                    section.append(line)
                if line==specific_phrase:
                    recording=True
            return section

    channels_of_interest = line_below("$ROI:", end="$PRESETS:")
    return {
    "data":         [float(line) for line in data if line.startswith(' ')],

    "date":         line_below("$DATE_MEA:")[0].split(" ")[0],
    "time":         line_below("$DATE_MEA:")[0].split(" ")[1], 

    "active time":  [int(i) for i in line_below("$MEAS_TIM:")[0].split(" ")][0],
    "run time":     [int(i) for i in line_below("$MEAS_TIM:")[0].split(" ")][1],
    "energy fit":         [float(i) for i in line_below("$ENER_FIT:" , end=2)[0].split(" ")], 
    "energy calibration": [float(i) for i in line_below("$MCA_CAL:"  , end=2)[-1].split(" ")[:3]],
    "shape calibration" : [float(i) for i in line_below("$SHAPE_CAL:", end=2)[-1].split(" ")],
    "channel range" : [int(i) for i in line_below("$DATA:", end=1)[0].split(" ")],
    "ROI":          [[int(i) for i in minmax.split(" ")] for minmax in channels_of_interest]
    }

def calculate_polynomial(inp, constants):
    return np.sum([constants[i]*inp**i for i in range(len(constants))], axis=0)

if __name__=='__main__':
    spec_dict = read_file('Bone4-c-12h-16-12-2019-m_A000.Spe')
    chan = np.arange(len(spec_dict['data']),step=1)
    p0, p1, p2 = spec_dict['energy calibration']
    P0, P1     = spec_dict['energy fit']
    E1 = calculate_polynomial(chan, [p0, p1, p2])
    E2 = calculate_polynomial(chan, [P0, P1])
    plt.semilogy(E1, spec_dict['data'])
    plt.xlabel("centroid energy of bin (keV)")
    plt.ylabel("counts per bin")
    plt.show()