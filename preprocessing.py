import geopandas as gp
import pandas as pd
import shapely as sp
import matplotlib.pyplot as plt
import numpy as np

crs=32650
def bar(process, dig=0, msg='Processing'):
    q = round(process * 100, dig)
    print(f'\r[{msg}:', "*" * round(q / 2), '_' * (50 - round(q / 2)), q, '%]', end="")


def mergeGdf(*roadFiles: str) -> gp.GeoDataFrame:
    roadGdf = gp.read_file(roadFiles[0]).to_crs(crs)
    for i, f in enumerate(roadFiles):
        if i > 0:
            rdf = gp.read_file(f).to_crs(crs)
            roadGdf = pd.concat([roadGdf, rdf])
    return roadGdf


def updateRoad(shp: gp.GeoDataFrame, speedTag='speed') -> pd.DataFrame:
    shp.index = range(len(shp))
    roadPoi = shp.get_coordinates()
    roadLOI = pd.DataFrame([], columns=['idx', 'name', 'speed', 'fromX', 'fromY', 'toX', 'toY', 'len'])
    currentIdx = -1
    for i, idx in enumerate(roadPoi.index):
        if idx > currentIdx:
            currentIdx = idx
        else:
            series = [
                idx,
                shp.loc[idx, 'NAME'],
                shp.loc[idx, speedTag],
                roadPoi.iloc[i, 0],
                roadPoi.iloc[i, 1],
                roadPoi.iloc[i - 1, 0],
                roadPoi.iloc[i - 1, 1],
                (roadPoi.iloc[i, 0] - roadPoi.iloc[i - 1, 0]) ** 2 + (roadPoi.iloc[i, 1] - roadPoi.iloc[i - 1, 1]) ** 2
            ]
            roadLOI.loc[len(roadLOI)] = series
        bar(i / len(roadPoi), 2, msg='BreakingPolyline')
    print(roadLOI)
    roadLOI.to_csv('roadLOI.csv')
    return roadLOI


def updatePOI(roadLOI: pd.DataFrame) -> pd.DataFrame:
    roadLOI['len'] = np.sqrt(roadLOI['len'])
    roadLOI['from'] = [roundLocation(x=x, y=y) for x, y in zip(roadLOI['fromX'], roadLOI['fromY'])]
    roadLOI['to'] = [roundLocation(x=x, y=y) for x, y in zip(roadLOI['toX'], roadLOI['toY'])]

    POI = np.unique(np.append(roadLOI['from'], roadLOI['to']))
    roadLOIReverse = roadLOI.copy()
    roadLOIReverse['from'] = roadLOI['to']
    roadLOIReverse['to'] = roadLOI['from']
    roadLOIReverse['fromX']=roadLOI['toX']
    roadLOIReverse['fromY'] = roadLOI['toY']
    roadLOI = pd.concat([roadLOI, roadLOIReverse]).groupby('from')
    POIConnection, POICost,realX,realY = [], [],[],[]
    for i, poi in enumerate(POI):
        con = roadLOI.get_group(poi)
        POIConnection.append('-'.join(con['to'].to_numpy()))
        POICost.append('-'.join(np.round((con['len'] / con['speed']), 2).astype(str)))
        realX.append(con['fromX'].iloc[0])
        realY.append(con['fromY'].iloc[0])
        bar(i / len(POI), 2, msg='NetworkTopology')
    POI = pd.DataFrame(np.array([POI, POIConnection, POICost,realX,realY]).T, columns=['poi', 'con', 'cost','rx','ry'])
    POI['x'], POI['y'] = np.array([p[1:-1].split(',') for p in POI['poi']]).T.astype(float)
    POI.to_csv('POI.csv')
    return POI


def roundLocation( x=None, y=None,location: str = None,toStr=True):
    if location is None:
        if y is None:
            return None
        else:
            location = [x, y]
    else:
        location = location[1:-1].split(',')
    location = np.array(location).astype(float).round(decimals=1).astype(str)
    if toStr:
        return f'({location[0][-7:]},{location[1][-7:]})'
    else:
        return np.array([location[0][-7:],location[1][-7:]]).astype(float)


def loadLOI(LOIFile='roadLOI.csv'):
    LOI = pd.read_csv(LOIFile,index_col=0)
    geometry = [sp.linestrings([roundLocation(fx,fy,toStr=False),roundLocation(tx,ty,toStr=False)]) for fx,fy,tx,ty in zip(LOI['fromX'],LOI['fromY'],LOI['toX'],LOI['toY'])]
    return gp.GeoDataFrame(LOI,geometry=geometry)

def loadPOI(POIFile='POI.csv'):
    if isinstance(POIFile, str):
        POI = pd.read_csv(POIFile,index_col=0)
    con,cost=[],[]
    for i in POI.index:
        con += [np.array(POI.loc[i, 'con'].split('-'))]
        cost += [np.array(POI.loc[i, 'cost'].split('-')).astype(float)]
        bar(i / len(POI), 2, msg='DecodeList')
    POI['con']=con
    POI['cost']=cost
    POI.sort_values(['x', 'y'], ascending=True, inplace=True)
    POI = gp.GeoDataFrame(POI,geometry=sp.points(POI['x'],POI['y']))
    return POI.set_index('poi')


def plotObject(*ojbect: sp.Geometry, show=True, **kwargs):
    for i, obj in enumerate(ojbect):
        coor = np.array(sp.get_coordinates(obj))
        plt.plot(coor[:,0],coor[:,1], **kwargs)
        bar(i / len(ojbect), 2, msg='Plotting...')
    if show:
        plt.show()


def preprocessSHP(*roadFiles: str) -> pd.DataFrame:
    gdf = mergeGdf(*roadFiles)
    loi = updateRoad(gdf)
    poi = updatePOI(loi)
    return loadPOI(poi)


if __name__ == '__main__':
    updatePOI(pd.read_csv('roadLOI.csv'))
    # preprocessSHP(r'shp\四级道-北京.shp',r'shp\三级道-北京.shp',r'shp\二级道-北京.shp',r'shp\一级道-北京.shp',r'shp\国道-北京.shp')

