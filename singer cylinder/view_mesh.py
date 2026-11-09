
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from params import CONFIG

def main():
 import argparse,json
 parser=argparse.ArgumentParser();parser.add_argument('--config');args=parser.parse_args()
 c=json.loads(Path(args.config).read_text(encoding='utf8')) if args.config else CONFIG
 p=Path(c['grid']['mesh_path']);d=np.load(p)
 xy=d['xy']; tri=d['triangles']
 edge=np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]))
 edge=np.unique(np.sort(edge,axis=1),axis=0)
 segments=xy[edge]
 g=c['grid'];wx0,wx1,wy0,wy1=c['physics']['wake_box']
 near=((-1,1),(-1,1)) if c['geometry']['name'].lower()=='cylinder' else ((-.35,1.35),(-.4,.4))
 for name,xlim,ylim in [('global',(g['xmin'],g['xmax']),(g['ymin'],g['ymax'])),('near',*near),('wake',(wx0,wx1),(wy0,wy1))]:
  fig,ax=plt.subplots(figsize=(12,5))
  ax.add_collection(LineCollection(segments,colors='#364f68',linewidths=.13,rasterized=True))
  ax.set_xlim(xlim);ax.set_ylim(ylim);ax.set_aspect('equal');ax.set_xlabel('x/c');ax.set_ylabel('y/c')
  ax.set_title(c['geometry']['name']+' — triangular unstructured mesh')
  fig.tight_layout();fig.savefig(p.with_name('mesh_'+name+'.png'),dpi=220);plt.close(fig)
 print('figures saved next to',p)
if __name__=='__main__':main()
