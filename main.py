import geopandas as gp
import pandas as pd
import shapely as sp
import numpy as np
import pyproj
from preprocessing import crs,roundLocation
from preprocessing import preprocessSHP,plotObject
from preprocessing import loadPOI,loadLOI


def Ax(start: str, end: str, POI: gp.GeoDataFrame) -> np.ndarray[str]:
    # def gx(now):
    #     origin = now
    #     g = 0
    #     while origin != start:
    #         for i, con in tempPOI.loc[origin, 'connection']:
    #             if tempPOI.loc[origin, 'target'] == con:
    #                 g += tempPOI.loc[origin, 'cost'][i]
    #         origin = tempPOI.loc[origin, 'target']
    #     return g

    def hx(now):
        h = sp.distance(tempPOI.loc[now, 'geometry'], tempPOI.loc[end, 'geometry'])
        return h

    tempPOI = POI.copy()
    tempPOI['target'] = None
    OPEN, CLOSE = {start: hx(start)}, {}
    NEXT = start

    def search(now):
        if now == end:
            return True
        CLOSE[now] = OPEN[now]
        del OPEN[now]
        for jx, tr in enumerate(tempPOI.loc[now, 'con']):
            ftr = CLOSE[now] - hx(now) + tempPOI.loc[now, 'cost'][jx] + hx(tr)
            if tr in CLOSE.keys():
                if ftr < CLOSE[tr]:
                    del CLOSE[tr]
                    OPEN[tr] = ftr
                    tempPOI.loc[tr, 'target'] = now
            elif tr in OPEN.keys():
                if ftr < OPEN[tr]:
                    OPEN[tr] = ftr
                    tempPOI.loc[tr, 'target'] = now
            else:
                OPEN[tr] = ftr
                tempPOI.loc[tr, 'target'] = now
        return False

    iters = 0
    while not search(NEXT):
        iters += 1
        _min = np.argmin(OPEN.values())
        NEXT = list(OPEN.keys())[_min]
        print(f'\rSTART:{start} END:{end} NEXT:{NEXT} iters:{iters}', end='')
    print('\r', end='')
    path = [end]
    while path[-1] != start:
        path.append(tempPOI.loc[path[-1], 'target'])
    return np.array(path).astype(str)

def modifiedAx(start: str, end: str, POI: gp.GeoDataFrame) -> list[str]:
    # def gx(now):
    #     origin = now
    #     g = 0
    #     while origin != start:
    #         for i, con in tempPOI.loc[origin, 'connection']:
    #             if tempPOI.loc[origin, 'target'] == con:
    #                 g += tempPOI.loc[origin, 'cost'][i]
    #         origin = tempPOI.loc[origin, 'target']
    #     return g

    def hx(now):
        h = sp.distance(tempPOI.loc[now, 'geometry'], tempPOI.loc[end, 'geometry'])
        return h

    tempPOI = POI.copy()
    tempPOI['target'] = None
    OPEN, CLOSE = {start: hx(start)}, {}
    fm = hx(start)
    NEXT = start

    def search(now):
        if now == end:
            return True
        CLOSE[now] = OPEN[now]
        del OPEN[now]
        for jx, tr in enumerate(tempPOI.loc[now, 'con']):
            ftr = CLOSE[now] - hx(now) + tempPOI.loc[now, 'cost'][jx] + hx(tr)
            if tr in CLOSE.keys():
                if ftr < CLOSE[tr]:
                    del CLOSE[tr]
                    OPEN[tr] = ftr
                    tempPOI.loc[tr, 'target'] = now
            elif tr in OPEN.keys():
                if ftr < OPEN[tr]:
                    OPEN[tr] = ftr
                    tempPOI.loc[tr, 'target'] = now
            else:
                OPEN[tr] = ftr
                tempPOI.loc[tr, 'target'] = now
        return False

    iters = 0

    while not search(NEXT):
        iters += 1
        _min = np.argmin(OPEN.values())
        NEXT = list(OPEN.keys())[_min]
        print(f'\rSTART:{start} END:{end} NEXT:{NEXT} iters:{iters}', end='')
    print('\r', end='')
    path = [end]
    while path[-1] != start:
        path.append(tempPOI.loc[end, 'target'])
    return path

def locate(lng: float, lat: float, POI: gp.GeoDataFrame) -> str:
    if lat < 180:
        transformer = pyproj.Transformer.from_crs(4326, crs)
        lng, lat = transformer.transform( lng, lat)
    lng, lat = roundLocation(lng, lat,toStr=False)
    absDist = np.abs(POI['x']-lng)+np.abs(POI['y']-lat)
    loc = POI.iloc[np.argmin(absDist)]
    return loc

def path2Polyline(poiList,POI)->sp.Geometry:
    poiList=POI.loc[poiList]
    return sp.linestrings(poiList[['rx','ry']].to_numpy())

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    POI = loadPOI()
    # LOI = loadLOI() # not required
    # plotObject(*LOI.geometry,color='black',alpha=0.2,show=False)
    print()
    ref=[54493.5,20736.3]
    start = locate(ref[0],ref[1],POI).name
    rands = (np.random.rand(100,2)-0.5)*1000
    end = [locate(ref[0]+r[0],ref[1]+r[1],POI).name for r in rands]
    pLs = []
    for ed in end:
        pLine=Ax(start,ed,POI)
        pLs.append(path2Polyline(pLine,POI))
    plotObject(*pLs,color='red')
    gp.GeoDataFrame(pd.DataFrame(range(len(pLs)),columns=['0']),geometry=pLs,crs=crs).to_file('roadBeijing.shp')

