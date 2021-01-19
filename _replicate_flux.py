from flux_convert import *


if __name__=="__main__":
    os.chdir("ChipIR_sparse")
    pd.read_csv('continuous_apriori.csv')
    tab = pd.read_csv('continuous_apriori.csv')

    continuous_apriori = tabulate(dict(zip(["x","y","interpolation"], tab.values.T)))

    gs = pd.read_csv('gs.csv')
    gs_ary = gs.values

    I = Integrate(continuous_apriori) #.definite_integral(*gs_ary.T)
    I.definite_integral(*gs_ary.T)
    gs_mock = np.logspace(np.log10(1/40), np.log10(70E6), 3000)
    gs_mock_ary = ary([gs_mock[:-1], gs_mock[1:]]).T

    fine_bin_int_flux = Integrate(continuous_apriori).definite_integral(*gs_mock_ary.T)
    per_eV_flux = flux_conversion(fine_bin_int_flux, gs_mock_ary, "integrated", "per eV")
    PUL_flux = flux_conversion(fine_bin_int_flux, gs_mock_ary, "integrated", "PUL")

    plt.loglog(gs_mock_ary.flatten(), np.repeat(per_eV_flux, 2)); plt.title(" per eV flux")
    plt.show()

    # the integration was the problem
    plt.loglog(gs_mock_ary.flatten(), np.repeat(per_eV_flux, 2)); plt.title(" per eV flux")
    plt.show()

    plt.loglog(gs_mock_ary.flatten(), np.repeat(PUL_flux, 2)); plt.title("Lethargy plot")
    plt.show()
