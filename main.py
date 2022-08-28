from obdet import obdet
from fr_analysis import analyzer


def perform_obdet_analysis():
    """
    Perform the OBDET analysis on the data.
    """
    args = obdet.get_params()
    obdet.perform_analysis(args)


if __name__ == '__main__':
    perform_obdet_analysis()
