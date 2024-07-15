import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation

from B_spline.predict_mse import predict_mse
from B_spline.bspline_coeffs import bspline_coeffs
from B_spline.preserve import preserve


def compress(sig, order, prd_limit, init_knot, show):
    x = np.arange(1, len(sig[0]) + 1)
    # print(sig)

    psig = preserve(x, sig[0])
    # print(psig)

    knots = init_knot
    not_knots = []
    fig, ax=plt.subplots(2, height_ratios=[1,3])
    c, prd, s, Rl, zl = bspline_coeffs(psig, knots, order)

    step = 1
    # print('prd', prd)
    # print('prd_lim', prd_limit)
    #plt.subplot(2, 2, 1)
    while prd < prd_limit:
        wpp = np.zeros(len(knots) - 2)
        for i in range(2, len(knots)):
            wpp[i - 2] = predict_mse(psig, s, c, order, knots, i)

        # print('wpp: ',wpp)
        indd = np.argmin(wpp[1:-1]) + 1
        not_knot = knots[indd + 1]
        # print(not_knot)
        tt = knots
        # print(tt)
        knots = np.delete(knots, indd + 1)
        """
        print("prd", prd)
        print("ind", indd + 1)
        print(wpp[indd + 1])
        # print(knots)
        print("psig")
        print(psig)
        print("knots")
        print(knots)
        print("order")
        print(order)
        print("indd")
        print(indd+1)
        print("tt")
        print(tt)
        print("Rl")
        print(Rl)
        print("zl")
        print(zl)
        """
        c, prd, s, Rl, zl = bspline_coeffs(
            psig, knots, order, indd + 1, tt, Rl, zl, show
        )

        if show:
            not_knots.append(not_knot)
            ax[0].clear()
            #ax[1].clear()
            width = np.max(psig) - np.min(psig)
            bl = np.min(s) - 0.1 * width
            ax[0].plot(
                not_knots,
                np.ones(len(not_knots)) * bl,
                "bx",
                markersize=12,
                linewidth=2,
            )
            ax[0].plot(knots, np.ones(len(knots)) * bl, "r.", markersize=5)
            ax[0].legend(
                [
                    "Original signal",
                    f"CR: {100 * (len(c) + len(knots)) / len(sig):.0f}%, PRD: {prd:.02f}%",
                ]
            )
            #ax[0].axis(
            #    [0, len(sig), np.min(psig) - 0.2 * width, np.max(psig) + 0.1 * width]
            #)
            plt.title(f"Iteration {step}")
            plt.draw()
            plt.pause(0.01)
            """
            """
            step += 1

    coeff = c

    if show:
        #plt.show()
        plt.figure()
        #plt.subplot(2, 2, 2)
        c, prd, s, _, _ = bspline_coeffs(psig, knots, order)
        plt.plot(psig, "b", linewidth=4)
        plt.plot(s, "r", linewidth=2)
        width = np.max(s) - np.min(s)
        bl = np.min(s) - 0.1 * width
        plt.plot(
            not_knots,
            np.ones(len(not_knots)) * bl,
            "bx",
            knots,
            np.ones(len(knots)) * bl,
            "r.",
            markersize=12,
            linewidth=2,
        )
        plt.stem(knots, psig[knots-1], "r.")
        plt.legend(["Original signal", f"PRD: {prd:.02f}%"])
        plt.axis([0, len(sig), np.min(psig) - 0.2 * width, np.max(psig) + 0.1 * width])
        plt.title("Approximation")

        plt.show()

    return s, knots, coeff, prd
