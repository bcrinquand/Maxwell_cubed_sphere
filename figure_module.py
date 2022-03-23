import matplotlib.pyplot as P
import matplotlib as mpl
from math import sqrt
P.ion()

tex_params = {
	"pgf.texsystem":   "pdflatex",       # Change this if using xetex or lautex
	"text.usetex":     True,             # Use LaTeX to write all text
	"font.family":     "serif",
	"axes.labelsize": 11,
	"axes.titlesize" : 11,
	"font.size": 11,
    # Make the legend/label fonts a little smaller
	"legend.fontsize": 9,
	"xtick.labelsize": 9,
	"ytick.labelsize": 9,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}", # Use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",     # Plots will be generated using this preamble
					]
}

P.rcParams.update(tex_params)

colours = [[0,0.55,0.55] , [0.86,0.08,0] , [1.0, 0.75, 0.0] , [0.52, 0.52, 0.51] , [0,0,0]]
marks  = ['o','^','s','D','<']
styles  = ['solid','dotted','dashed','dashdot']

from mpl_toolkits.axes_grid1 import make_axes_locatable, Size
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

golden_mean = (sqrt(5.0)-1.0)/2.0

# Returns figure size
def deffigsize(width_to_fig=0.5,height_to_width=golden_mean): 

	#number=1 if one plot on fig_width, 2 if two plots, 3 if three plots...
	#ratio: aspect ratio width of figure / height of figure

	fig_width_pt = 441.01775*width_to_fig           # get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # convert pt to inch
	fig_width = fig_width_pt*inches_per_pt          # width in inches
	fig_height = fig_width*height_to_width          # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

# Defines some plot parameters
def defrcparams(grid_bool):

	mpl.rc('axes',grid=grid_bool)
	mpl.rc('grid',linestyle='--')
	mpl.rc('lines',linewidth=1.5)
	mpl.rc('legend',frameon='False',numpoints=1)
	mpl.rc('xtick.minor',visible=True)
	mpl.rc('ytick.minor',visible=True)

# Saves figure
def figsave_png(filename,res=200):
    P.savefig('{}.png'.format(filename),bbox_inches = 'tight',pad_inches  = 0.05, dpi=res)

def figsave_pdf(filename):
    P.savefig('{}.pdf'.format(filename),bbox_inches = 'tight',pad_inches  = 0.05)

