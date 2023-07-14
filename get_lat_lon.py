#!/usr/bin/env python3

import mgrs
from typing import Tuple


def get_lat_lon(x_local: float, y_local: float) -> Tuple[float, float]:
    mgrs_grid = "54SUE"
    mgrs_string = mgrs_grid + str(int(x_local)).zfill(5) + str(int(y_local)).zfill(5)
    m = mgrs.MGRS()
    lat, lon = m.toLatLon(mgrs_string)
    return lat, lon


if __name__ == "__main__":
    x_local = 81377.35148449386
    y_local = 49916.90332548532
    lat, lon = get_lat_lon(x_local, y_local)
    print(f"{lat}, {lon}")
