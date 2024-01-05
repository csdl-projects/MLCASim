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
parser.add_argument('-lr', '--load_ratio', required=False, type=int, default = '2', help='number')
parser.add_argument('-si', '--sample_int', required=False, type=int, default = '10', help='number')

args = parser.parse_args()

number_scanline_pixel = args.number_scanline_pixel
number_dataline_pixel = args.number_dataline_pixel
res = args.res
cap = args.cap
tw_s = 'TH*2'
if tw_s == 1:
    args.tw_s = 'TH*1'
VDH = args.VDH
si = args.sample_int
load_ratio = args.load_ratio

parameters = {
    'C_DRG_DRS_R': '(334.866f) * loadratio',
    'C_DRG_SCAN_R': '(0.752f) * loadratio',
    'C_DRG_DATA_R': '(0.742f) * loadratio',
    'C_DRG_VDD_R': '(0.667f) * loadratio',
    'C_DRG_VREF_R': '(0.001f) * loadratio',
    'C_DRG_VSS_R': '(0.762f) * loadratio',
    'R_SCAN_R': '(0.622) * loadratio',
    'R_DATA_R': '(2.33) * loadratio',
    'DR_W_R': '24u',
    
    'C_DRG_DRS_G': '(338.839f) * loadratio',
    'C_DRG_SCAN_G': '(0.752f) * loadratio',
    'C_DRG_DATA_G': '(0.764f) * loadratio',
    'C_DRG_VDD_G': '(0.649f) * loadratio',
    'C_DRG_VREF_G': '(0.001f) * loadratio',
    'C_DRG_VSS_G': '(0.749f) * loadratio',
    'R_SCAN_G': '(0.622) * loadratio',
    'R_DATA_G': '(2.294) * loadratio',
    'DR_W_G': '24u',

    'C_DRG_DRS_B': '(232.275f) * loadratio',
    'C_DRG_SCAN_B': '(0.791f) * loadratio',
    'C_DRG_DATA_B': '(0.761f) * loadratio',
    'C_DRG_VDD_B': '(0.497f) * loadratio',
    'C_DRG_VREF_B': '(0.038f) * loadratio',
    'C_DRG_VSS_B': '(0.896f) * loadratio',
    'R_SCAN_B': '(0.622) * loadratio',
    'R_DATA_B': '(2.41) * loadratio',
    'DR_W_B': '22u',

    'C_DRG_DRS_W': '(266.271f) * loadratio',
    'C_DRG_SCAN_W': '(0.794f) * loadratio',
    'C_DRG_DATA_W': '(0.745f) * loadratio',
    'C_DRG_VDD_W': '(0.587f) * loadratio',
    'C_DRG_VREF_W': '(0.032f) * loadratio',
    'C_DRG_VSS_W': '(0.900f) * loadratio',
    'R_SCAN_W': '(0.622) * loadratio',
    'R_DATA_W': '(2.344) * loadratio',
    'DR_W_W': '18u',

    'R_VDD': '(0.742)',
    'R_VREF': '(1.454)',
}

if __name__ == '__main__':        
    # Open the SPICE netlist file for writing
    pixel_dir = '/project/common/LGD/spice_data/circuit'
    data_dir = '/project/common/LGD/spice_data/output'
    name = f'{number_scanline_pixel}_{number_dataline_pixel}_{res}_{cap}_{args.tw_s}_{VDH}_{load_ratio}'
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
.INC '../Base_OTV_PARFILE_Hspice_modify'
.INC '../schematic.sp'

*************************************************************************
.savebias ./Initial.par tran time = 280u  ALL
\n''')
        file.write('''
**********   <Load Parameter>   ******************************************
''')
        # Write the parameter variations
        for param, value in parameters.items():
            file.write('.PARAM {} = {}\n'.format(param, value))

        file.write('''\n
*************************************************************************
***                                                                   ***
***                               <SPICE Param>                       ***
***                                                                   ***
*************************************************************************
\n''')
        
        file.write('**************Scan Signal **************************\n')
        file.write(f".PARAM  T00  = 'TH*8 - TW_S - TMG'\n")
        for j in range(number_dataline_pixel):
            file.write(f".PARAM  T{j+1:02}  = 'T{j:02} + TH'\n")

        file.write('**************Scan left Signal **************************\n')
        for j in range(number_dataline_pixel):
            file.write(f"VSCAN_L<{j+1}> SCAN_L<{j+1}> 0 PULSE(GVGL VGH T{j+1:02} TR TF TW_S '1/FR')\n")

        file.write('**************Scan right Signal **************************\n')
        for j in range(number_dataline_pixel):
            file.write(f"VSCAN_R<{j+1}> SCAN_R<{j+1}> 0 PULSE(GVGL VGH T{j+1:02} TR TF TW_S '1/FR')\n")

        # Write the header section
        file.write('''\n
*************************************************************************
***                                                                   ***
***                               <Schematic>                         ***
***                                                                   ***
*************************************************************************
\n''')

        # Write the parameter variations for each iteration
        for j in range(number_dataline_pixel):
            for i in range(number_scanline_pixel):                
                id = j*number_scanline_pixel + i + 1
                next_id = (j + 1)*number_scanline_pixel + i + 1
                scanid = j*(number_scanline_pixel + 1) + i + 1
                file.write(f'X1<{id}> DATAR<{next_id}> DATAW<{next_id}> DATAB<{next_id}> DATAG<{next_id}> DATAR<{id}> DATAW<{id}> DATAB<{id}> DATAG<{id}> SCAN<{scanid}> SCAN<{scanid+1}> VDD \n+ VDD VREF<{j+2}> VREF<{j+1}> VSS unit_pxl_ver01\n')
        
        
        for j in range(number_dataline_pixel):
            right = (j+1)*(number_scanline_pixel+1)
            file.write(f'X2<{j+1}> SCAN_R<{j+1}> SCAN<{right}> scan_load_scale_top \n')
        
        for j in range(number_dataline_pixel):
            left = j*(number_scanline_pixel+1) + 1
            file.write(f'X3<{j+1}> SCAN_L<{j+1}> SCAN<{left}> scan_load_scale_top\n')

        for i in range(number_scanline_pixel):                
            id = i + 1
            file.write(f'X4<{4*i+1}> DATA_RU<{id}> DATAR<{id}> data_load_scale_top\n')
            file.write(f'X4<{4*i+2}> DATA_WU<{id}> DATAW<{id}> data_load_scale_top\n')
            file.write(f'X4<{4*i+3}> DATA_BU<{id}> DATAB<{id}> data_load_scale_top\n')
            file.write(f'X4<{4*i+4}> DATA_GU<{id}> DATAG<{id}> data_load_scale_top\n')

        for i in range(number_scanline_pixel):                
            id = i + 1
            file.write(f'X5<{4*i+1}> DATA_R<{id}> DATA_RU<{id}> data_load_pcb_top\n')
            file.write(f'X5<{4*i+2}> DATA_W<{id}> DATA_WU<{id}> data_load_pcb_top\n')
            file.write(f'X5<{4*i+3}> DATA_B<{id}> DATA_BU<{id}> data_load_pcb_top\n')
            file.write(f'X5<{4*i+4}> DATA_G<{id}> DATA_GU<{id}> data_load_pcb_top\n')

        for j in range(number_dataline_pixel):
            file.write(f'X6<{j+1}> VREF VREF<{j+1}> ref_load_scale_top\n')

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
''')
        file.write(f'.PARAM VDHW = {VDH}\n')
        file.write(f'.PARAM VDHR = {VDH}\n')
        file.write(f'.PARAM VDHG = {VDH}\n')
        file.write(f'.PARAM VDHB = {VDH}\n')
        file.write(f'.PARAM	loadratio = {load_ratio}\n')
        file.write('''           


.TRAN	20n '1/FR/2'

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

        # file.write(f'.PRINT v(scan_l) v(scan_r) \n')
        # file.write(f'.PRINT v(scan<1>) \n')
        # for i in range(si, number_scanline_pixel, si):
        #     file.write(f'.PRINT v(scan<{i}>) \n')
        # file.write(f'.PRINT v(scan<{number_scanline_pixel+1}>)\n') 
        
        ## WHITE
        for i in range(1, number_scanline_pixel*number_dataline_pixel+1, si):
            file.write(f'.PRINT V(X1<{i}>.XWHITE.drg) \n')

        for i in range(1, number_scanline_pixel*number_dataline_pixel+1, si):
            file.write(f'.PRINT V(X1<{i}>.XWHITE.drs) \n')

        ## WHITE
        for i in range(1, number_scanline_pixel*number_dataline_pixel+1, si):
            file.write(f'.PRINT I(X1<{i}>.XWHITE.XEL.d1) \n')
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
    