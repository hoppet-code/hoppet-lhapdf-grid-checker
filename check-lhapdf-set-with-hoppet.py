#! /usr/bin/env python3

"""
This is small tool that takes an LHAPDF grid as input and then checks
if the evolution agrees with that of hoppet. It takes a number of
commandline arguments but by default it will read the PDF at the
lowest scale and evolve it up. It outputs a number of plots and some
basic information to the terminal.

Example commandline: python check-lhapdf-set.py -pdf NNPDF30_nnlo_as_0118 
"""

import hoppet as hp
import lhapdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import datetime
import sys
from concurrent.futures import ProcessPoolExecutor
from PyPDF2 import PdfMerger
import time
import math


# Colour codes
PURPLE = '\033[95m'
ORANGE = '\033[93m'
RED = '\033[91m'
GREEN = '\033[92m'
END = '\033[0m'

eps = 1e-4 # Epsilon for crossing mass thresholds. Has to be set globally since it is used in a number of different functions

def main():
    # First get the command line arguments
    args = get_commandline()

    matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing safety
    hoppet_ver = version() # Hoppet version for plots

    # This routine loads an lhapdf set at its initial scale (unless otherwise specified) and evolves it with hoppet.
    p_lhapdf, Q0, output, mc, mb, mt, xmin, xmax, Qmin, Qmax = load_lhapdf_start_evolve_hoppet(args)

    # Evaluate the PDFs at some x values and print them
    xvals = [1e-5,1e-4,1e-3,1e-2,0.1,0.3,0.5,0.7,0.9]
    Q = 100.0
    
    # Threshold for coloring and heatmap (red if above threshold green if below)
    threshold = args.prec_threshold

    print(f"Using threshold for relative deviation: {PURPLE}{threshold*100:.2f}%{END}\nRelative deviation printed in parentheses: {RED}red{END} if above, {GREEN}green{END} if below\n")

    print("")
    print(f"                                                   Evaluating PDFs at Q = {Q} GeV")
    print(f"                          hoppet evolution with rel. deviation in parentheses |(hoppet - lhapdf)/hoppet|")
    print(f"    {PURPLE}x            u-ubar                 d-dbar               2(ubr+dbr)               c+cbar                 gluon{END}")
    for ix in range(9):
        x = xvals[ix]
        pdf_hoppet = hp.Eval(x, Q)
        pdf_lhapdf = np.zeros(13)
        pdf_lhapdf[0+6] = p_lhapdf.xfxQ(21, x, Q)
        for flavor in range(-6, 7):
            if flavor == 0:
                continue
            pdf_lhapdf[flavor + 6] = p_lhapdf.xfxQ(flavor, x, Q)
     # Calculate values and deviations
        vals = [
            (pdf_hoppet[6 + 2] - pdf_hoppet[6 - 2], pdf_lhapdf[6 + 2] - pdf_lhapdf[6 - 2]),
            (pdf_hoppet[6 + 1] - pdf_hoppet[6 - 1], pdf_lhapdf[6 + 1] - pdf_lhapdf[6 - 1]),
            (2 * (pdf_hoppet[6 - 1] + pdf_hoppet[6 - 2]), 2 * (pdf_lhapdf[6 - 1] + pdf_lhapdf[6 - 2])),
            (pdf_hoppet[6 - 4] + pdf_hoppet[6 + 4], pdf_lhapdf[6 - 4] + pdf_lhapdf[6 + 4]),
            (pdf_hoppet[6 + 0], pdf_lhapdf[6 + 0])
        ]
        out_str = f"{x:7.1E} "
        for hoppet_val, lhapdf_val in vals:
            # Avoid division by zero
            if abs(hoppet_val) > 0:
                rel_dev = abs((lhapdf_val - hoppet_val) / hoppet_val)
            else:
                rel_dev = 0.0
            color = RED if abs(rel_dev) > threshold else GREEN
            out_str += f"{hoppet_val:11.4E} ({color}{rel_dev:7.2E}{END}) "
        print(out_str)

    # Threshold transition checks for mc and mb
    print("\nThreshold transition checks:")
    header = f"{PURPLE}{'x':>5}  {'charm<mc':>20}  {'charm>mc':>22}  {'bottom<mb':>22}  {'bottom>mb':>22}{END}"
    print(header)
    for x in xvals:
        # charm
        hoppet_c_below = hp.Eval(x, mc - eps)[6 + 4]
        lhapdf_c_below = p_lhapdf.xfxQ(4, x, mc - eps)
        rel_c_below = abs((lhapdf_c_below - hoppet_c_below) / hoppet_c_below) if abs(hoppet_c_below) > 0 else 0.0
        color_c_below = RED if rel_c_below > threshold else GREEN
        charm_below_str = f"{hoppet_c_below:11.4E} ({color_c_below}{rel_c_below:7.2E}{END})"

        hoppet_c_above = hp.Eval(x, mc + eps)[6 + 4]
        lhapdf_c_above = p_lhapdf.xfxQ(4, x, mc + eps)
        rel_c_above = abs((lhapdf_c_above - hoppet_c_above) / hoppet_c_above) if abs(hoppet_c_above) > 0 else 0.0
        color_c_above = RED if rel_c_above > threshold else GREEN
        charm_above_str = f"{hoppet_c_above:11.4E} ({color_c_above}{rel_c_above:7.2E}{END})"

        # bottom
        hoppet_b_below = hp.Eval(x, mb - eps)[6 + 5]
        lhapdf_b_below = p_lhapdf.xfxQ(5, x, mb - eps)
        rel_b_below = abs((lhapdf_b_below - hoppet_b_below) / hoppet_b_below) if abs(hoppet_b_below) > 0 else 0.0
        color_b_below = RED if rel_b_below > threshold else GREEN
        bottom_below_str = f"{hoppet_b_below:11.4E} ({color_b_below}{rel_b_below:7.2E}{END})"

        hoppet_b_above = hp.Eval(x, mb + eps)[6 + 5]
        lhapdf_b_above = p_lhapdf.xfxQ(5, x, mb + eps)
        rel_b_above = abs((lhapdf_b_above - hoppet_b_above) / hoppet_b_above) if abs(hoppet_b_above) > 0 else 0.0
        color_b_above = RED if rel_b_above > threshold else GREEN
        bottom_above_str = f"{hoppet_b_above:11.4E} ({color_b_above}{rel_b_above:7.2E}{END})"

        print(f"{x:8.2e}  {charm_below_str:>20}  {charm_above_str:>20}  {bottom_below_str:>20}  {bottom_above_str:>20}")

    # Print alphas comparison between HOPPET and LHAPDF
    print("\nαS comparison:")
    Qvals_alpha = [1.0, mc+eps, 3.0, mb+eps, 5.0, 50.0, 91.1876, 100.0, 500.0, 1000.0, Qmax]
    print(f"{PURPLE}{'Q':>7}  {'HOPPET':>16}  {'LHAPDF':>12}  {'|rel. dev.|':>13}{END}")
    for Qval in Qvals_alpha:
        alphas_hoppet = hp.AlphaS(Qval)
        alphas_lhapdf = p_lhapdf.alphasQ(Qval)
        rel_dev = abs((alphas_lhapdf - alphas_hoppet) / alphas_hoppet) if abs(alphas_hoppet) > 0 else 0.0
        color = RED if rel_dev > threshold else GREEN
        print(f"{Qval:11.4f}  {alphas_hoppet:12.6f}  {alphas_lhapdf:12.6f}  {color}{rel_dev:12.4e}{END}")
    print("")

    print_deviations_plot_heatmaps(args, Q0, mc, mb, mt, threshold, output, hoppet_ver, p_lhapdf, hp, max(xmin,args.xmin), min(xmax,args.xmax), Qmin, min(Qmax,args.Qmax), args.nbins, args.do_plots)
    print("Output saved to file:   ", f"{PURPLE}{output}.txt{END}")
    print("")

    # Cleanup just for good measure
    hp.DeleteAll()

def load_lhapdf_start_evolve_hoppet(args):
    # Load the pdf from LHAPDF
    pdf = args.pdf
    p_lhapdf = lhapdf.mkPDF(args.pdf, 0)

    # Now that we have the PDF we define the interface as needed by hoppet
    def lhapdf_interface(x, Q):
        pdf = np.zeros(13)
        lhapdf = p_lhapdf.xfxQ(None, x, Q)
        # Map HOPPET indices to LHAPDF PIDs
        pid_map = [ -6, -5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5, 6 ]
        for i, pid in enumerate(pid_map):
            pdf[i] = lhapdf.get(pid, 0.0)
        return pdf

    # Get some information from the PDF like order in QCD, masses etc.
    nloop = p_lhapdf.orderQCD + 1 # LHAPDF starts at 0
    xmin = p_lhapdf.xMin
    xmax = p_lhapdf.xMax
    Qmin = np.sqrt(p_lhapdf.q2Min)
    Qmax = np.sqrt(p_lhapdf.q2Max)
    mc = p_lhapdf.quarkThreshold(4)
    mb = p_lhapdf.quarkThreshold(5)
    mt = p_lhapdf.quarkThreshold(6)
    if(p_lhapdf.hasFlavor(6) == False): mt = 2*Qmax# If no top is defined set it to a high value
    
    # By default we evolve from Qmin but -Q0 can be specified by the user
    Q0 = max(args.Q0, Qmin)
    if args.Q0_just_above_mc:
        Q0 = mc + eps
    elif args.Q0_just_above_mb:
        Q0 = mb + eps
    asQ0 = p_lhapdf.alphasQ(Q0)

    # Define output name
    output = f"{pdf}_Q0{Q0}_hoppet_check"
    sys.stdout = Tee(f"{output}.log")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print some info to the screen
    print(f"{pdf} read succesfully with the following parameters extracted: \nNumber of loops: {nloop}\nxmin: {xmin}\nxmax: {xmax}\nQmin: {Qmin}\nQmax: {Qmax}\nmc: {mc}\nmb: {mb}\nmt: {mt}")

    # Now we start hoppet
    dy = args.dy
    
    # By default we use parametrised nf thresholds and splitting
    # functions (this only applies to the NNLO part, since at N3LO we
    # are currently forced to use exact nf but approximate splitting
    # functions).
    hp.SetExactDGLAP(args.exact_nnlo_nf, args.exact_nnlo_splitting)
    print(f"Using exact NNLO nf thresholds: {args.exact_nnlo_nf}, exact NNLO splitting functions: {args.exact_nnlo_splitting}")

    # n3lo splitting function approximation
    if nloop == 4:
        if args.n3lo_splitting == '2310':
            hp.SetApproximateDGLAPN3LO(hp.n3lo_splitting_approximation_up_to_2310_05744)
        elif args.n3lo_splitting == '2404':
            hp.SetApproximateDGLAPN3LO(hp.n3lo_splitting_approximation_up_to_2404_09701)
        elif args.n3lo_splitting == '2410':
            hp.SetApproximateDGLAPN3LO(hp.n3lo_splitting_approximation_up_to_2410_08089) # This is the default value in hoppet at the moment
        else:
            print(f"{RED}Error: Unknown n3lo-splitting value '{args.n3lo_splitting}'{END}")
            sys.exit(1)
        print(f"N3LO splitting function approximation: {args.n3lo_splitting}")
        
    # Right now I can't see a way to find the scheme in the LHAPDF
    # interface. For now we assume it is variable unless the user
    # specifies FFN on the commandline
    if args.FFN > 0:
        hp.SetFFN(args.FFN)
        print(f"Using Fixed Flavour Number scheme with nf = {args.FFN}")
    else:
        hp.SetPoleMassVFN(mc,mb,mt)
        print(f"Using Pole Mass Variable Flavour Number scheme with mc = {mc}, mb = {mb}, mt = {mt}")

    #hp.Start(dy, nloop)
    ymax = float(math.ceil(np.log(1.0/xmin)))
    if ymax > 15.0:
        dlnlnQ = dy/8.0
    else:
        dlnlnQ = dy/4.0
    
    hp.SetYLnlnQInterpOrders(args.yorder, args.lnlnQorder)
    print(f"Set yorder = {args.yorder}, lnlnQorder = {args.lnlnQorder}")
    print(f"Starting Hoppet with ymax = {ymax} and dy = {dy} and nloop = {nloop} and dlnlnQ = {dlnlnQ} and order = {args.order}")
    hp.StartExtended(ymax, dy, Qmin, Qmax, dlnlnQ, nloop, args.order, hp.factscheme_MSbar)

    # If the PDF uses a truncated solution to the renormalization
    # group equation then reading the coupling at the low scale can
    # lead to differences
    if(args.alphasQ0 > 0.0): 
        asQ0 = p_lhapdf.alphasQ(args.alphasQ0)
        hp.SetCoupling(asQ0, args.alphasQ0, nloop)
        asQ0 = hp.AlphaS(Q0)
        #hp.DeleteAll()


    
    # Bit of a workaround right now that we always call the evolution
    # routine, because otherwise no coupling gets set up.
    if args.assign:
        print(f"Assigning PDF using hoppetAssign using Q0 = {Q0} GeV with as(Q0) = {asQ0}")
        hp.SetCoupling(asQ0, Q0, nloop)
        hp.Assign(lhapdf_interface)
    else:
        print(f"Evolving PDF from Q0 = {Q0} GeV with as(Q0) = {asQ0}")
        hp.Evolve(asQ0, Q0, nloop, 1.0, lhapdf_interface, Q0)

    return p_lhapdf, Q0, output, mc, mb, mt, xmin, xmax, Qmin, Qmax # This is a bit strange, but the only way to make sure that we first set it inside this function and then return it

def get_commandline():
    # Get commandline
    parser = argparse.ArgumentParser(description="Check of an LHAPDF grid against HOPPET evolution.")
    parser.add_argument('-pdf', required=True, help='LHAPDF set name (required, ex. NNPDF30_nnlo_as_0118)')
    parser.add_argument('-dy', type=float, default=0.05, help='dy for HOPPET evolution (default: 0.05)')
    parser.add_argument('-order', type=int, default=-6, help='Order for HOPPET evolution (default: -6)')
    parser.add_argument('-yorder', type=int, default=5, help='yorder for HOPPET evolution (default: -1)')
    parser.add_argument('-lnlnQorder', type=int, default=4, help='lnlnQorder for HOPPET evolution (default: 4)')
    parser.add_argument('-prec-threshold', type=float, default=5e-3, help='Threshold for relative deviation, used for colour coding and summary analysis (default: 5e-3)')
    parser.add_argument('-nbins', type=int, default=100, help='Number of bins for heatmap in log(x) and log(Q) (default: 100)')
    parser.add_argument('-Q0', type=float, default=1.0, help='Initial Q0 value (default: Qmin from LHAPDF)')
    parser.add_argument('-alphasQ0', type=float, default=-1.0, help='Initial Q0 value to be used in alphas, by default it uses Q0')
    parser.add_argument('-Q0-just-above-mc', action='store_true', help='Set Q0 just above mc')
    parser.add_argument('-Q0-just-above-mb', action='store_true', help='Set Q0 just above mb')
    parser.add_argument("-FFN", type=int, default=-1, help="Fixed flavour number scheme with nf=FFN. Negative values will result in the variable flavour number scheme being used (default: -1)")
    parser.add_argument('-exact-nnlo-nf', type=lambda x: x.lower() not in ['false','0','no'], default=False, nargs='?', const=True, help='Use exact nf thresholds at NNLO (default: False)')
    parser.add_argument('-exact-nnlo-splitting', type=lambda x: x.lower() not in ['false','0','no'], default=False, nargs='?', const=True, help='Use exact splitting functions at NNLO (default: False)')
    parser.add_argument('-n3lo-splitting', type=str, default ="2410", help='N3LO splitting function approximation (see n3lo_splitting_approximation flag in dglap_choices.f90 in hoppet for explanation) (2310, 2404, 2410 (default))')
    parser.add_argument('-do-plots', type=lambda x: x.lower() not in ['false','0','no'], default=False, nargs='?', const=True, help='Enable plotting of 2D heatmaps. Takes O(10s) for nbins=100. (default: False)')
    parser.add_argument('-xmax', type=float, default=0.9,  help='Maximum x value for plots and summary (default: 0.9)')
    parser.add_argument('-xmin', type=float, default=1e-5, help='Minimum x value for plots and summary (default: 1e-5)')
    parser.add_argument('-Qmax', type=float, default=28e3, help='Maximum Q value for plots and summary (default: 28e3 GeV)')
    parser.add_argument('-njobs', type=int, default=None, help='Number of parallel jobs for plotting (default: use all available cores)')
    parser.add_argument('-blind', action='store_true', help='Enable blind mode, ie do not print PDF info on plots (default: False)')
    parser.add_argument('-assign', action='store_true', help='Assign the PDF directly to hoppet instead of evolving it (default: False)')
    args = parser.parse_args()

    return args

# Plotting script for 2D heatmaps. Plots the relative deviations
# between lhapdf and hoppet, using hoppet as the baseline. 
def print_deviations_plot_heatmaps(args, Q0, mc, mb, mt, threshold, output, hoppet_ver, p_lhapdf, hp, xmin, xmax, Qmin, Qmax, nbins, do_plots):
    xvals = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
    Qvals = np.logspace(np.log10(Qmin), np.log10(Qmax), nbins)
    flavors = range(-5, 6)  # -5 to 5, gluon is 0
    info_str = f"PDF: {args.pdf}   Q0: {Q0:.3f} GeV  dy: {args.dy}   mc: {mc:.3f} GeV  mb: {mb:.3f} GeV"
    if(args.blind): info_str = f"PDF: ???  Q0: {Q0:.3f} GeV  dy: {args.dy}   mc: {mc:.3f} GeV  mb: {mb:.3f} GeV"
    xticks = generate_xticks(xmin, xmax)
    bounds = [0.0, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 1e0]
    colors = ['darkgreen', 'green', 'limegreen', 'greenyellow', 'yellow', 'orange', 'red', 'brown', 'black']
    #colors = ['darkgreen', 'green', 'darkblue', 'blue', 'yellow', 'orange', 'red', 'brown', 'black']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Precompute all hoppet and lhapdf values for all (Q, x)
    # Shape: (len(Qvals), len(xvals), 13)
    print("Filling hoppet and lhapdf arrays...\n")
    hoppet_all = np.zeros((len(Qvals), len(xvals), 13))
    lhapdf_all = np.zeros((len(Qvals), len(xvals), 13))
    for i, Q in enumerate(Qvals):
        for j, x in enumerate(xvals):
            hoppet_all[i, j, :] = hp.Eval(x, Q)
            lhapdf_dict = p_lhapdf.xfxQ(None, x, Q)
            for flavor in range(-6, 7):
                if flavor == 0:
                    lhapdf_all[i, j, flavor + 6] = lhapdf_dict.get(21, 0.0)
                else:
                    lhapdf_all[i, j, flavor + 6] = lhapdf_dict.get(flavor, 0.0)

    # Vectorized deviation and bin counting
    mask = np.abs(hoppet_all) > 0
    deviation = np.zeros_like(hoppet_all)
    deviation[mask] = np.abs((lhapdf_all[mask] - hoppet_all[mask]) / hoppet_all[mask])
    flat_dev = deviation.flatten()
    bin_indices = np.digitize(flat_dev, bounds) - 1
    valid = (bin_indices >= 0) & (bin_indices < len(bounds) - 1)
    bin_counts = np.bincount(bin_indices[valid], minlength=len(bounds)-1)
    total_points = valid.sum()

    print("Summary of points within a certain deviation from hoppet (across all flavors):")
    print("Kinematic range: xmin = ", xmin, ", xmax = ", xmax, ", Qmin = ", Qmin, "GeV, Qmax = ", Qmax, "GeV")
    print("Bin range      Percentage")
    for i in range(len(bin_counts)):
        lower = bounds[i]
        upper = bounds[i+1]
        percent = 100.0 * bin_counts[i] / total_points if total_points > 0 else 0.0
        left = percent_label(lower)
        right = percent_label(upper) if upper != 1e0 else ""
        bin_str = f"{left} - {right}" if right else f"{left} -"
        colour = RED if threshold < upper else GREEN
        print(f"{bin_str:<14} {colour}{percent:8.2f}%{END}")
    print("")

    # Region-wise deviation checks for PDF
    region_names = [f"Q < mc ({mc:.2f})", f"mc ≤ Q < mb ({mb:.2f})", f"mb ≤ Q < mt ({mt:.2f})", f"Q ≥ mt ({mt:.2f})"]
    region_counts = []
    region_total = []
    Qarr = np.array(Qvals)
    for ireg, (low, high) in enumerate([(None, mc), (mc, mb), (mb, mt), (mt, None)]):
        if low is None:
            maskQ = Qarr < high
        elif high is None:
            maskQ = Qarr >= low
        else:
            maskQ = (Qarr >= low) & (Qarr < high)
        region_dev = deviation[maskQ, :, :]
        region_total.append(region_dev.size)
        region_counts.append(np.sum(region_dev >= threshold))

    print(f"Fraction of points with deviation >= {PURPLE}{threshold*100:.2f}%{END} in each Q region (PDFs):")
    for idx, name in enumerate(region_names):
        percent = 100.0 * region_counts[idx] / region_total[idx] if region_total[idx] > 0 else 0.0
        if percent > 1.0:
            colour = RED
            status = f"{ORANGE}CHECK PLOTS{END}"
        else:
            colour = GREEN
            status = f"{GREEN}OK{END}"
        print(f"  {name:<29}: {colour}{percent:6.2f}%{END} {status}")

    # Region-wise deviation checks for alphaS
    Qvals_alpha_plot = np.logspace(np.log10(Qmin), np.log10(Qmax), nbins*10)
    alphas_hoppet = np.array([hp.AlphaS(Q) for Q in Qvals_alpha_plot])
    alphas_lhapdf = np.array([p_lhapdf.alphasQ(Q) for Q in Qvals_alpha_plot])
    rel_dev_alpha = np.abs((alphas_lhapdf - alphas_hoppet) / alphas_hoppet)
    region_counts_alpha = []
    region_total_alpha = []
    Qarr_alpha = Qvals_alpha_plot
    for low, high in [(None, mc), (mc, mb), (mb, mt), (mt, None)]:
        if low is None:
            maskQ = Qarr_alpha < high
        elif high is None:
            maskQ = Qarr_alpha >= low
        else:
            maskQ = (Qarr_alpha >= low) & (Qarr_alpha < high)
        region_dev = rel_dev_alpha[maskQ]
        region_total_alpha.append(region_dev.size)
        region_counts_alpha.append(np.sum(region_dev >= threshold))

    print(f"\nFraction of points with deviation >= {PURPLE}{threshold*100:.2f}%{END} in each Q region (αS):")
    for idx, name in enumerate(region_names):
        percent = 100.0 * region_counts_alpha[idx] / region_total_alpha[idx] if region_total_alpha[idx] > 0 else 0.0
        if percent > 1.0:
            colour = RED
            status = f"{ORANGE}CHECK PLOTS{END}"
        else:
            colour = GREEN
            status = f"{GREEN}OK{END}"
        print(f"  {name:<29}: {colour}{percent:6.2f}%{END} {status}")
    print("")

    # Check if plots should be generated
    if not do_plots:
        print(f"{PURPLE}Plotting disabled.{END} Enable with -do-plots True.")
        return
    
    print(f"Plotting deviation heatmaps for {args.pdf} with nbins = {nbins}. This could take a while... (disable with -do-plots False).\n")
    # Use ProcessPoolExecutor to parallelize flavor plots
    args_list = [
        (flavor, hoppet_all, lhapdf_all, xvals, Qvals, norm, cmap, bounds, info_str, xticks, output)
        for flavor in flavors
    ]
    with ProcessPoolExecutor(max_workers=args.njobs) as executor:
        tmpfiles = list(executor.map(plot_flavor, args_list))

    # Merge all temporary PDFs into the final output PDF
    merger = PdfMerger()
    for tmpfile in tmpfiles:
        merger.append(tmpfile)

    # Extra plots for mass thresholds (charm and bottom)
    mass_tmpfiles = []
    for flavor, mass, label in [(4, mc, 'charm'), (5, mb, 'bottom')]:
        Qvals_linear = np.linspace(mass, 4*mass, nbins)
        deviation = np.zeros((len(Qvals_linear), len(xvals)))
        for i, Q in enumerate(Qvals_linear):
            for j, x in enumerate(xvals):
                hoppet_val = hp.Eval(x, Q)[flavor + 6]
                lhapdf_val = p_lhapdf.xfxQ(flavor, x, Q)
                if abs(hoppet_val) > 0:
                    rel_diff = abs((lhapdf_val - hoppet_val) / hoppet_val)
                else:
                    rel_diff = 0.0
                deviation[i, j] = rel_diff

        plt.figure(figsize=(8, 6))
        X, Y = np.meshgrid(xvals, Qvals_linear)
        plt.pcolormesh(X, Y, deviation, norm=norm, shading='auto', cmap=cmap)
        cbar = plt.colorbar(label='|(hoppet - lhapdf) / hoppet|', boundaries=bounds, ticks=bounds)
        cbar.ax.set_yticklabels([percent_label(b) for b in bounds])
        plt.xscale('log')
        plt.xlabel('x')
        plt.ylabel('Q [GeV]')
        plt.title(f'Mass threshold region ({label}): flavor {flavor}')
        plt.figtext(0.5, 0.01, info_str, ha='center', va='bottom', fontsize=10, color='grey', alpha=0.5)
        plt.figtext(0.98, 0.98, f"hoppet v{hoppet_ver}", ha='right', va='top', fontsize=10, color='grey', alpha=0.5)
        plt.xticks(xticks, [format_xtick(tick) for tick in xticks])
        plt.tight_layout()
        tmpfile = f"{output}_mass_{label}_{flavor}.pdf"
        with PdfPages(tmpfile) as pdf_pages:
            pdf_pages.savefig()
        plt.close()
        mass_tmpfiles.append(tmpfile)
    for tmpfile in mass_tmpfiles:
        merger.append(tmpfile)

    # Add plot for alphas relative deviation
    Qvals_alpha_plot = np.logspace(np.log10(Qmin), np.log10(Qmax), nbins*10)
    alphas_hoppet = np.array([hp.AlphaS(Q) for Q in Qvals_alpha_plot])
    alphas_lhapdf = np.array([p_lhapdf.alphasQ(Q) for Q in Qvals_alpha_plot])
    rel_dev_alpha = np.abs((alphas_lhapdf - alphas_hoppet) / alphas_hoppet)
    plt.figure(figsize=(8, 6))
    plt.plot(Qvals_alpha_plot, rel_dev_alpha)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Q [GeV]')
    plt.ylabel('Relative deviation in alphaS')
    plt.title('Relative deviation in alphaS vs Q')
    plt.figtext(0.5, 0.01, info_str, ha='center', va='bottom', fontsize=10, color='grey', alpha=0.5)
    plt.figtext(0.98, 0.98, f"hoppet v{hoppet_ver}", ha='right', va='top', fontsize=10, color='grey', alpha=0.5)
    plt.tight_layout()
    tmpfile = f"{output}_alphas.pdf"
    with PdfPages(tmpfile) as pdf_pages:
        pdf_pages.savefig()
    plt.close()
    merger.append(tmpfile)

    merger.write(f"{output}.pdf")
    merger.close()
    # Optionally, clean up temporary files
    import os
    for tmpfile in tmpfiles + mass_tmpfiles + [f"{output}_alphas.pdf"]:
        os.remove(tmpfile)
    print("Plots saved to PDF file:", f"{PURPLE}{output}.pdf{END}")

# Object to handle output to both terminal and a file
class Tee(object):
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def plot_flavor(args):
    (flavor, hoppet_all, lhapdf_all, xvals, Qvals, norm, cmap, bounds, info_str, xticks, output) = args
    hoppet_vals = hoppet_all[:, :, flavor + 6]
    lhapdf_vals = lhapdf_all[:, :, flavor + 6]
    mask = np.abs(hoppet_vals) > 0
    deviation = np.zeros_like(hoppet_vals)
    deviation[mask] = np.abs((lhapdf_vals[mask] - hoppet_vals[mask]) / hoppet_vals[mask])
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(xvals, Qvals)
    plt.pcolormesh(X, Y, deviation, norm=norm, shading='auto', cmap=cmap)
    cbar = plt.colorbar(label='|(hoppet - lhapdf) / hoppet|', boundaries=bounds, ticks=bounds)
    cbar.ax.set_yticklabels([percent_label(b) for b in bounds])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel('Q [GeV]')
    plt.title(f'|Rel. deviation| for flavor {flavor}')
    plt.figtext(0.5, 0.01, info_str, ha='center', va='bottom', fontsize=10, color='grey', alpha=0.5)
    hoppet_ver = version()
    plt.figtext(0.98, 0.98, f"hoppet v{hoppet_ver}", ha='right', va='top', fontsize=10, color='grey', alpha=0.5)
    plt.xticks(xticks, [format_xtick(tick) for tick in xticks])
    plt.tight_layout()
    tmpfile = f"{output}_flavor_{flavor}.pdf"
    with PdfPages(tmpfile) as pdf_pages:
        pdf_pages.savefig()
    plt.close()
    return tmpfile

def generate_xticks(xmin, xmax):
    xticks = [xmin]
    # Find the nearest lower power of 10 to xmin
    log10_xmin = math.ceil(math.log10(xmin))
    # Start from 1eN, where N = log10_xmin
    tick = 10 ** log10_xmin
    while tick < xmax:
        if tick > xmin:
            xticks.append(tick)
        tick *= 10
    if xmin > 5e-6:
        xticks.extend([0.3, 0.5, 0.9])
    else:
        xticks.extend([0.3, 0.9])
    return xticks

def format_xtick(tick):
    if tick == 0:
        return "0"
    log10 = np.log10(tick)
    if np.isclose(log10, int(log10)):
        return f"$10^{{{int(log10)}}}$"
    elif tick < 1e-2:
        return f""
    else:
        return f"{tick:.1f}"
    
def percent_label(b):
    if b >= 1e0:
        return '   '
    else:
        return f'{b*100:.2f}%'

def version():
    import subprocess
    try:
        out = subprocess.check_output(['hoppet-config', '--version'], universal_newlines=True)
        return out.strip()
    except Exception:
        return "unknown"

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds.\n")

