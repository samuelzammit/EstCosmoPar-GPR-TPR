# based on https://github.com/lucavisinelli/H0TensionRealm

# import required packages/modules
import csv
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

black = [0, 0, 0]
red = [1, 0, 0]
blue = [0, 0, 1]

# rcParams['mathtext.fontset'] = 'custom'
# rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

class ErrorLinePlotter:
	def __init__(self, data, position):
		self.data = data
		self.position = position

		# Horizontal line
		self.hlwidth = 0.8
		self.hlstyle = '-'
		self.hlcolor = red

		# Point props
		self.point_size = 0.34
		self.point_color = blue
		self.point_lwidth = 0.8

		self.middle_point_type = 'line'
		self.middle_point_size = self.point_size
		self.middle_point_color = self.point_color

		self.middle_point_lwidth = self.point_lwidth
		self.middle_point_mshape = 'o'

	def set_props(self, hlwidth, hlstyle, hlcolor,
				  psize, pcolor, pwidth,
				  middle_point_type='line',
				  **lmprop):

		self.hlwidth = hlwidth
		self.hlstyle = hlstyle
		self.hlcolor = hlcolor
		self.point_size = psize
		self.point_color = pcolor
		self.point_lwidth = pwidth

		self.middle_point_size = psize
		self.middle_point_color = pcolor
		self.middle_point_type = middle_point_type
		if middle_point_type == 'line':
			if len(lmprop) != 0:
				self.middle_point_size = lmprop['mpsize']
				self.middle_point_color = lmprop['mpcolor']
				self.middle_point_lwidth = lmprop['lwidth']
		elif middle_point_type == 'marker':
			if len(lmprop) != 0:
				self.middle_point_size = lmprop['mpsize']
				self.middle_point_color = lmprop['mpcolor']
				self.middle_point_mshape = lmprop['mshape']

	def plot(self):
		list_3 = [self.data['ml'] + self.data['e1_sig'][0],
				  self.data['ml'],
				  self.data['ml'] + self.data['e1_sig'][1]]

		plt.hlines(y=self.position, xmin=list_3[0], xmax=list_3[-1], color=self.hlcolor, linestyles=self.hlstyle,
				   lw=self.hlwidth, zorder=1)
		plt.vlines(x=[list_3[0], list_3[-1]],
				   ymin=self.position - self.point_size / 2,
				   ymax=self.position + self.point_size / 2,
				   color=self.point_color,
				   linestyles='-',
				   lw=self.point_lwidth,
				   zorder=2)

		if self.middle_point_type == 'line':
			plt.vlines(x=list_3[1],
					   ymin=self.position - self.middle_point_size / 2,
					   ymax=self.position + self.middle_point_size / 2,
					   color=self.middle_point_color,
					   ls='-',
					   lw=self.middle_point_lwidth,
					   zorder=3)
		elif self.middle_point_type == 'marker':
			plt.scatter(list_3[1], self.position,
						s=self.middle_point_size,
						color=self.middle_point_color,
						marker=self.middle_point_mshape,
						zorder=3)
		else:
			print("Error: Invalid middle point type.")
			sys.exit()


# repository containing the .csv with the dataset
data_path = "G:/My Drive/MSc/SOR5200 - Dissertation/GaPP_27/Application/6_Comparison/comparison_datasets/"


# sigma distance between two values of H0: see eq. (28) of Briffa et al. (2020)
def calcsigmadistance(h01, h02, sigma1, sigma2):
	dist = (h01 - h02) / np.sqrt(sigma1**2 + sigma2**2)
	if dist >= 0:
		# paras.append(det[i] + str(value[i]) + '${\pm}$' + str(upper[i]))
		dist = " " + "{:.4f}".format(round(dist, 4))
	else:
		dist = "{:.4f}".format(round(dist, 4))
	# dist = "{:.4f}".format(round(dist, 4))
	return dist


def plotComparison(methodname='all', datasetname='all', priorname='all', doublesize=False, printvalues=False):
	# fil = data_path + dataname + 'Dataset.csv'
	fil = data_path + "AveragesDataset_noHomGPTP.csv"

	# load the dataset and count the number of data points
	nr = 1
	with open(fil, 'r+') as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			nr += 1

	# load the data points into arrays
	# unlike in "6_comparison_new.py" here we have only one "name" field
	# instead of method, prior, dataset
	value = np.zeros(nr)
	lower = np.zeros(nr)
	upper = np.zeros(nr)
	name = ["" for x in range(nr)]

	i = 0
	with open(fil, 'r+') as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			name[i] = row[0]
			value[i] = float(row[1])
			lower[i] = float(row[2])
			upper[i] = float(row[3])
			i += 1

	# remove last entry as reader is somehow reading an extra blank row
	temp = len(name)-1
	name = name[:temp]
	value = value[:temp]
	lower = lower[:temp]
	upper = upper[:temp]

	# number of rows filtered
	nrnew = len(name)

	# join name with value
	paras = []
	for i in range(nrnew):
		if printvalues:
			if lower[i] == upper[i]:
				paras.append(name[i] + str(value[i]) + '${\pm}$' + str(upper[i]))
			else:
				paras.append(name[i] + str(value[i]) + '$^{+' + str(upper[i]) + '}_{-' + str(lower[i]) + '}$')
		else:
			paras.append(name[i])
	# data
	all_data = []
	# print(value)
	# print(upper)
	# print(lower)
	# print(nrnew)
	for i in range(nrnew):
		all_data.append({'ml': value[i], 'e1_sig': [upper[i], -lower[i]]})

	# add average
	meanvalue = np.mean(value)
	meanupper = np.mean(upper)
	meanlower = np.mean(lower)
	if printvalues:
		if meanlower == meanupper:
			paras.append("Average: " + str(meanvalue) + '${\pm}$' + str(meanupper))
		else:
			paras.append("Average: " + str(meanvalue) + '$^{+' + str(meanupper) + '}_{-' + str(meanlower) + '}$')
	else:
		paras.append("Average: ")
	all_data.append({'ml': meanvalue, 'e1_sig': [meanupper, -meanlower]})

	# style
	pos_num = nrnew + 1  # + 1 to include average

	positions = []
	labels = []
	for i in range(pos_num + 2):
		positions.append(i)
		labels.append('')

	# move to "files_paper" directory
	os.chdir("G:/My Drive/MSc/SOR5200 - Dissertation/GaPP_27/Application/6_Comparison/files_paper")

	# plot
	# pdf = PdfPages('H0whisker_GPRsorted.pdf')

	# get filename for plot
	pdf = PdfPages('H0whisker_Averages.pdf')

	if doublesize:
		plt.rcParams['figure.figsize'] = (8, 10)
		# plt.text(78, nrnew - 1, "${H_0}\,$[km$\,$s$^{-1}\,$Mpc$^{-1}$]", size=9)
	else:
		plt.rcParams['figure.figsize'] = (4, 5)
		# plt.text(73, nrnew + 1.7, "${H_0}\,$[km$\,$s$^{-1}\,$Mpc$^{-1}$]", size=7)

	# plot the vertical bars for reference: R20 vs CMB
	plt.bar(74.22, 100, width=1.82, facecolor='cyan', alpha=0.15)  # Riess (2019)
	plt.bar(67.4, 100, width=0.5, facecolor='pink', alpha=0.25)	# Planck (2020)

	# plot each data point with attached label
	ypos = 0
	for i in range(len(paras)):
		ypos = nrnew - i + 1
		elp = ErrorLinePlotter(all_data[i], position=ypos)
		labels[elp.position] = paras[i]
		if i != len(paras) - 1:  # for individual readings
			plt.text(55.5, ypos - 0.1, calcsigmadistance(value[i], 67.4, upper[i], 0.5), size=5)
			plt.text(80, ypos - 0.1, calcsigmadistance(value[i], 74.22, upper[i], 1.82), size=5)
		else:					# for average at the end
			plt.text(55.5, ypos - 0.1, calcsigmadistance(meanvalue, 67.4, meanupper, 0.5), size=5)
			plt.text(80, ypos - 0.1, calcsigmadistance(meanvalue, 74.22, meanupper, 1.82), size=5)
		elp.set_props(0.8, '-', black, 0.7, black, 0.8, 'marker', mpsize=2.0, mpcolor=black, mshape='o')
		elp.plot()

	# add dotted line between individual readings and average
	plt.axhline(y=ypos + 0.5, color='black', linewidth=0.5, linestyle='dashed')

	axis_label_size = 5.5
	plt.tick_params(axis='x', labelsize=axis_label_size+0.5)
	plt.tick_params(axis='y', labelsize=axis_label_size)
	plt.xticks([i for i in range(60, 85, 5)])  # ,fontweight='semibold')
	plt.xlim(55, 85)
	plt.ylim(positions[0], positions[-1])
	plt.yticks(positions, labels)
	plt.tight_layout()

	# x-axis label
	plt.xlabel("$H_0 \ (km \ s^{-1} \ Mpc^{-1})$", fontdict={'size': axis_label_size+1})

	pdf.savefig()
	plt.clf()
	plt.cla()
	plt.close()
	pdf.close()
	os.chdir("../")
	print("Plot done.")


# new plots
plotComparison()
