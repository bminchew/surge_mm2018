
def set_mpl(mpl):
    fntsize = 25
    labsize = fntsize + 8
    font = {'size'   : fntsize}
    axes = {'linewidth' : 1}
    tick = {'major.pad' : 12,
            'major.size' : 8,
            'minor.size' : 5}
    mathfont = {'fontset' : 'cm'}

    mpl.rc('font', **font)
    mpl.rc('axes', **axes)
    mpl.rc('xtick', **tick)
    mpl.rc('ytick', **tick)
    mpl.rc('mathtext', **mathfont)

    return mpl,fntsize,labsize
