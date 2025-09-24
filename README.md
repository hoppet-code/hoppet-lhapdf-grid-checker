# hoppet-lhapdf-grid-checker
This is a small python script that reads an
[LHAPDF](https://www.lhapdf.org/) grid and compares its internal
evolution to that of [hoppet](https://github.com/hoppet-code/hoppet).
The script prints a number of useful diagnostics to screen and
produces a set of plots to compare the two evolutions.

## Getting started
The code requires that both hoppet and LHAPDF's python interfaces have
been built and are visible to python (hoppet can also be installed
from PyPi with `pip install hoppet`). Assuming that is the case, and
that all other dependencies are met (if the script complains about a
missing package try `pip install package`) the code can simply be run
with

```
./check-lhapdf-set-with-hoppet.py -pdf LHAPDFname
```

To get a full list of commandline arguments run
```
./check-lhapdf-set-with-hoppet.py --help
``` 
For a standard user the only flag that is needed is `-pdf` and if
plots are required `-do-plots`. As an example, running

```
./check-lhapdf-set-with-hoppet.py -pdf PDF4LHC21_40 -do-plots
```

will initialise hoppet at the $Q_{\mathrm{min}}$ of the `PDF4LHC21_40`
set, and fill a grid using hoppet's evolution. This grid is then
compared to the LHAPDF grid across a large range of $x$ and $Q$ and
relative deviations are computed. If the script identifies regions
with large relative deviations (the threshold can be set with
`-prec-threshold` which by default is `5d-3`), they will be printed in
<span style="color:red">red</span> on screen. The results are printed
on screen and saved in `PDF4LHC21_40_Q01.4001_hoppet_check.txt`.

The plots can be found in `PDF4LHC21_40_Q01.4001_hoppet_check.pdf`.
The heatmaps should be green -- if not then there are regions with
deviations. Below you can as an example see the relative deviation
from hoppet for the gluon, across the full range of $Q$ and $x$, and
the bottom close to its production threshold.
<table>
<tr>
<td><img src="example/PDF4LHC21_40_Q01.4001_hoppet_check-06.png" alt="Image 1" width="400"></td>
<td><img src="example/PDF4LHC21_40_Q01.4001_hoppet_check-13.png" alt="Image 2" width="400"></td>
</tr>
</table>

Both the `.txt` and `.pdf` files can be found in the [example](example/) directory.

## Citation
Please cite https://arxiv.org/abs/0804.3755
(original v1 release) and https://arxiv.org/abs/2509.nnnnn (v2 features) if using the results of this script in a scientific publication.

## Contact and bugs
Please get in touch on
[alexander.karlberg@cern.ch](mailto:alexander.karlberg@cern.ch) for
any queries or bug reports (or directly on gitHub).