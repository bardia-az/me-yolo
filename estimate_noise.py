import argparse
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def Gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def read_y_channel(f, w, h):
    raw = f.read(w*h)
    raw = np.frombuffer(raw, dtype=np.uint8)
    # raw = raw.reshape((h,w))
    return raw.astype(np.int)

def estimate_noise(ref_file, target_file, frame_num):
    ref_f = open(ref_file, 'rb')
    trg_f = open(target_file, 'rb')
    w = h = 1024

    hist = np.zeros(511)
    for fr in tqdm(range(frame_num)):
        ref = read_y_channel(ref_f, w, h)
        trg = read_y_channel(trg_f, w, h)
        diff = ref - trg
        tmp_hist, bin_edges = np.histogram(diff, bins=range(-255,257), density=True)
        # print(bin_edges.size())
        # print(tmp_hist.size())
        # print(len(tmp_hist))
        # print(len(bin_edges[:-1]))
        # print(bin_edges[:-1])
        hist = tmp_hist + hist

    hist = hist / frame_num
    print(sum(hist))

    return hist
    
    # plt.bar(bin_edges[:-1], hist, width=1)
    # plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--QP"  ,      default = None, type=int)

    args = parser.parse_args()

    video_names = ['Kimono', 'ParkScene', 'PeopleOnStreet', 'Traffic']
    ref_names = ['class_b_Kimono_tensor_1024x1024.yuv', 'class_b_ParkScene_tensor_1024x1024.yuv', 'class_a_PeopleOnStreet_tensor_1024x1024.yuv', 'class_a_Traffic_tensor_1024x1024.yuv']
    frame_nums = [240, 240, 150, 150]

    target_files = ['../Latent_Motion_Estimation/working_dir/reconst_' + name + f'_QP{args.QP}.yuv' for name in video_names]
    # target_file = '../Latent_Motion_Estimation/working_dir/reconst_Kimono_QP34.yuv'
    ref_files = ['../TO_BARDIA/' + refs for refs in ref_names]
    # ref_file = '../TO_BARDIA/class_b_Kimono_tensor_1024x1024.yuv'
    # frame_num = 240
    # print(target_files)
    # print(ref_files)

    pdf = np.zeros(511)
    for i in range(len(video_names)):
        print(target_files[i])
        print(ref_files[i])
        print(frame_nums[i])
        tmp_pdf = estimate_noise(ref_files[i], target_files[i], frame_nums[i])
        pdf = tmp_pdf + pdf
        
    pdf = pdf/len(target_files)
    print(sum(pdf))

    print(f'probability of 0 is {pdf[255]:.3f}')
    print(f'probability of 1 is {pdf[256]:.3f}')
    print(f'probability of -1 is {pdf[254]:.3f}')
    print(f'probability of 2 is {pdf[257]:.3f}')
    print(f'probability of -2 is {pdf[253]:.3f}')

    # np.save(f'Noise_Distributions/QP_{args.QP}.npy', pdf)

    x = range(-255,256)
    plt.bar(x, pdf, width=1)
    plt.show()

    popt, pcov = curve_fit(Gauss, x, pdf)

    # print(len(popt))
    print(f'amplitude is \t {popt[0]}')
    print(f'mean is \t {popt[1]}')
    print(f'std is \t {popt[2]}')

    fit_y = Gauss(x, popt[0], popt[1], popt[2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, pdf, label='pdf')
    ax.plot(x, fit_y, c='r', label='Gaussian fit')
    ax.legend()
    plt.show()
    

if __name__ == '__main__':
    main()


