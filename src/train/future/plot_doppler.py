import os
import numpy as np
import matplotlib.pyplot as plt


def plot_doppler(npz_path, out_png=None, vmax_db=0.0, vmin_db=-30.0):
    data = np.load(npz_path)
    S = data['S']  # [windows, doppler_bins]
    delta_v = float(data.get('delta_v', np.nan))
    Tc = float(data.get('Tc', np.nan))
    N_D = S.shape[1] if S.size else 0

    if S.size == 0:
        raise ValueError('Empty Doppler spectrum in NPZ')

    # Convert to dB for visualization
    SdB = 10.0 * np.log10(np.maximum(S, 1e-12))

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(SdB.T, cmap='viridis', vmin=vmin_db, vmax=vmax_db)
    im.set_edgecolor('face')
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('normalized power [dB]')

    ax.set_xlabel('time windows')

    # Y-axis: velocity using delta_v if available
    if np.isfinite(delta_v) and N_D > 0:
        length_v = N_D // 2
        step = max(10, length_v // 6)
        ticks_y = np.arange(length_v - step * (length_v // step), length_v + step * (length_v // step) + 1, step)
        ax.set_yticks(ticks_y + 0.5)
        ax.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
        ax.set_ylabel('velocity [m/s]')
    else:
        ax.set_ylabel('doppler bin')

    ax.set_title('Doppler spectrum')

    plt.tight_layout()
    if out_png is None:
        base = os.path.splitext(npz_path)[0]
        out_png = base + '_plot.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    return out_png


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='Path to doppler_spectrum.npz')
    parser.add_argument('--out', help='Output PNG path', default=None)
    parser.add_argument('--vmax', type=float, default=0.0)
    parser.add_argument('--vmin', type=float, default=-30.0)
    args = parser.parse_args()

    out = plot_doppler(args.npz, out_png=args.out, vmax_db=args.vmax, vmin_db=args.vmin)
    print('Saved plot to', out)
