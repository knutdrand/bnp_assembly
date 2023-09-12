from bnp_assembly import plotting
import os


def test_report():
    plotting_folder = "testplotting"
    if os.path.isfile(plotting_folder + "/report.html"):
        os.remove(plotting_folder + "/report.html")

    plotting.register(testplotting=plotting.ResultFolder(plotting_folder))

    px = plotting.px(name="testplotting")
    px.txt("Test 123", title="Testfile")
    px.write_report()

    assert os.path.isfile(plotting_folder + "/report.html")

    # test subplotter
    sub = px.sublogger("sublogger")
    sub.txt("Test 1234", title="Testfile2")
    px.write_report()

    assert os.path.isfile(plotting_folder + "/sublogger/report.html")

