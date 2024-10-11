import healpy as hp
import sys
import matplotlib.pyplot as plt

# enter the two mask files as arguments. Must be healpix maps.
mask1_file = sys.argv[1]
mask2_file = sys.argv[2]
# optionally enter a tag for the figure name§
if len(sys.argv) > 3:
    tag = sys.argv[3]
else:
    tag = mask1_file.split('/')[-1].split('.')[:-1] + '_' + mask2_file.split('/')[-1].split('.')[:-1] # this will prob throw an error
#print(mask1_file.split('/')[-1].split('.')[:-1] + '_' + mask2_file.split('/')[-1].split('.')[:-1])

mask1 = hp.read_map(mask1_file)
mask2 = hp.read_map(mask2_file)
mask = mask1 * mask2

hp.mollview(mask)
plt.savefig(f'{tag}.png')
hp.write_map(''.join(mask1_file.split('/')[:-1])+f'{tag}.fits', mask)