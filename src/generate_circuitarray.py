import subprocess as sp
import os
import time
from argparse import ArgumentParser

parser = ArgumentParser(description='SPICE')
parser.add_argument('-s', '--number_scanline_pixel', required=True, type=int, default = '10', help='number')
parser.add_argument('-d', '--number_dataline_pixel', required=True, type=int, default = '10', help='number')
parser.add_argument('-r', '--res', required=False, type=float, default = '8.0', help='number')
parser.add_argument('-c', '--cap', required=False, type=float, default = '0.008', help='number')
parser.add_argument('-t', '--tw_s', required=False, type=int, default = '2', help='number')
parser.add_argument('-v', '--VDH', required=False, type=float, default = '5.8', help='number')
parser.add_argument('-sr', '--sample_int', required=False, type=int, default = '10', help='number')

args = parser.parse_args()

number_scanline_pixel = args.number_scanline_pixel
number_dataline_pixel = args.number_dataline_pixel
res = args.res
cap = args.cap
tw_s = 'TH*2'
if tw_s == 1:
    args.tw_s = 'TH*1'
VDH = args.VDH
sr = args.sample_int

parameters = {
    'C_DRG_DRS_R': '(334.866f)',
    'C_DRG_SCAN_R': '(0.752f)',
    'C_DRG_DATA_R': '(0.742f)',
    'C_DRG_VDD_R': '(0.667f)',
    'C_DRG_VREF_R': '(0.001f)',
    'C_DRG_VSS_R': '(0.762f)',
    'R_SCAN_R': '(0.622)',
    'R_DATA_R': '(2.33)',
    'DR_W_R': '24u',
    
    'C_DRG_DRS_G': '(338.839f)',
    'C_DRG_SCAN_G': '(0.752f)',
    'C_DRG_DATA_G': '(0.764f)',
    'C_DRG_VDD_G': '(0.649f)',
    'C_DRG_VREF_G': '(0.001f)',
    'C_DRG_VSS_G': '(0.749f)',
    'R_SCAN_G': '(0.622)',
    'R_DATA_G': '(2.294)',
    'DR_W_G': '24u',

    'C_DRG_DRS_B': '(232.275f)',
    'C_DRG_SCAN_B': '(0.791f)',
    'C_DRG_DATA_B': '(0.761f)',
    'C_DRG_VDD_B': '(0.497f)',
    'C_DRG_VREF_B': '(0.038f)',
    'C_DRG_VSS_B': '(0.896f)',
    'R_SCAN_B': '(0.622)',
    'R_DATA_B': '(2.41)',
    'DR_W_B': '22u',

    'C_DRG_DRS_W': '(266.271f)',
    'C_DRG_SCAN_W': '(0.794f)',
    'C_DRG_DATA_W': '(0.745f)',
    'C_DRG_VDD_W': '(0.587f)',
    'C_DRG_VREF_W': '(0.032f)',
    'C_DRG_VSS_W': '(0.900f)',
    'R_SCAN_W': '(0.622)',
    'R_DATA_W': '(2.344)',
    'DR_W_W': '18u',

    'R_VDD': '(0.742)',
    'R_VREF': '(1.454)',
}

if __name__ == '__main__':        
    # Open the SPICE netlist file for writing
    pixel_dir = '/project/common/LGD/spice_data/circuit'
    data_dir = '/project/common/LGD/spice_data/output'
    name = f'{number_scanline_pixel}_{number_dataline_pixel}_{res}_{cap}_{args.tw_s}_{VDH}'
    sp_name = f'Array{name}.sp'
    lis_name = f'Array{name}.lis'
    with open(os.path.join(pixel_dir, sp_name), 'w') as file:
        # Write the header section
        file.write('''\n
*************************************************************************
***                                                                   ***
***                               < OTV >                             ***
***                                                                   ***
*************************************************************************

**********   <Include File>   *******************************************
.INC '../OTV_PARFILE_Hspice_modify'
.INC '../schematic.sp'

*************************************************************************
.option numdgt = 8
.savebias ./Initial.par tran time = 280u  ALL
\n''')
        file.write('''
**********   <Load Parameter>   ******************************************
.PARAM  C_DRG_DRS_W = '(266.271f) * loadratio'
.PARAM  C_DRG_SCAN_W = '(0.794f) * loadratio'
.PARAM  C_DRG_DATA_W = '(0.745f) * loadratio'
.PARAM  C_DRG_VDD_W = '(0.587f) * loadratio'
.PARAM  C_DRG_VREF_W = '(0.032f) * loadratio'
.PARAM  C_DRG_VSS_W = '(0.900f) * loadratio'

.PARAM  R_SCAN_W = '( 0.622) * loadratio'

.PARAM  R_DATA_W = '( 2.344)'

.PARAM  R_VDD = '( 0.742)'

.PARAM  R_VREF = '( 1.454)'

.PARAM  DR_W_W   = '18u'
''')

        # Write the parameter variations
        for param, value in parameters.items():
            file.write('.PARAM {} = {}\n'.format(param, value))

        # Write the header section
        file.write('''\n
*************************************************************************
***                                                                   ***
***                               <Schematic>                         ***
***                                                                   ***
*************************************************************************
\n''')

        # Write the parameter variations for each iteration
        for i in range(number_scanline_pixel):
            file.write(f'X1<{i+1}> NET1<{4*i}> NET1<{4*i+1}> NET1<{4*i+2}> NET1<{4*i+3}> DATA DATA DATA DATA SCAN<{i+1}> SCAN<{i+2}> VDD \n+ VDD NET2<{i+1}> VREF VSS unit_pxl_ver01\n')
            
        file.write(f'X1 SCAN_L SCAN<1> scan_load_scale_top\n')
        file.write(f'X2 SCAN_R SCAN<{number_scanline_pixel+1}> scan_load_scale_top \n')
        file.write('''\n
*************************************************************************
***                                                                   ***
***                                  <Run>                            ***
***                                                                   ***
*************************************************************************
\n''')
        file.write('''
.PARAM  Px = 1920
.PARAM  Py = 100
.PARAM	VDHW = 5.8
.PARAM	loadratio = 2

.TRAN	20n 300u

******************** Print node *****************************************

\n''')
        # file.write(f'.PROBE v(scan_l) v(scan_r) \n')
        # file.write(f'.PROBE v(scan<1>) \n')
        # for i in range(100, number_scanline_pixel, 100):
        #     file.write(f'.PROBE v(scan<{i}>) \n')
        # file.write(f'.PROBE v(scan<{number_scanline_pixel+1}>\n') 
        
        # ## WHITE
        # file.write(f'.PROBE V(X1<1>.XWHITE.drg) \n')
        # for i in range(100, number_scanline_pixel, 100):
        #     file.write(f'.PROBE V(X1<{i}>.XWHITE.drg) \n')
        # file.write(f'.PROBE V(X1<{number_scanline_pixel}>.XWHITE.drg) \n')

        # file.write(f'.PROBE V(X1<1>.XWHITE.drs) \n')
        # for i in range(100, number_scanline_pixel, 100):
        #     file.write(f'.PROBE V(X1<{i}>.XWHITE.drs) \n')
        # file.write(f'.PROBE V(X1<{number_scanline_pixel}>.XWHITE.drs) \n')

        # ## WHITE
        # file.write(f'.PROBE I(X1<1>.XWHITE.XEL.d1) \n')
        # for i in range(100, number_scanline_pixel, 100):
        #     file.write(f'.PROBE I(X1<{i}>.XWHITE.XEL.d1) \n')
        # file.write(f'.PROBE I(X1<{number_scanline_pixel}>.XWHITE.XEL.d1)) \n')

        file.write(f'.print v(scan_l) v(scan_r) \n')
        file.write(f'.print v(scan<1>) \n')
        for i in range(sr, number_scanline_pixel, sr):
            file.write(f'.print v(scan<{i}>) \n')
        file.write(f'.print v(scan<{number_scanline_pixel+1}>)\n') 
        
        ## WHITE
        file.write(f'.print V(X1<1>.XWHITE.drg) \n')
        for i in range(sr, number_scanline_pixel, sr):
            file.write(f'.print V(X1<{i}>.XWHITE.drg) \n')
        file.write(f'.print V(X1<{number_scanline_pixel}>.XWHITE.drg) \n')

        file.write(f'.print V(X1<1>.XWHITE.drs) \n')
        for i in range(sr, number_scanline_pixel, sr):
            file.write(f'.print V(X1<{i}>.XWHITE.drs) \n')
        file.write(f'.print V(X1<{number_scanline_pixel}>.XWHITE.drs) \n')

        ## WHITE
        file.write(f'.print isub(X1<1>.XWHITE.XEL.d1) \n')
        for i in range(sr, number_scanline_pixel, sr):
            file.write(f'.print isub(X1<{i}>.XWHITE.XEL.d1) \n')
        file.write(f'.print isub(X1<{number_scanline_pixel}>.XWHITE.XEL.d1) \n')
        file.write('''
.GLOBAL GND
*************************************************************************

.END
\n''')
    def ExecuteCommand( curCmd ):
        print( curCmd )
        sp.call( curCmd, shell=True)
    s = time.time()
    ExecuteCommand(f'hspice -i {os.path.join(pixel_dir, sp_name)} -o {os.path.join(data_dir, lis_name)}')
    print(f"{name} : TIME {time.time() - s:.2f}")
    