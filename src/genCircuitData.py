##################################################################
## Code to generate the SPICE netlist for the Pixel3T1C circuit ##
##################################################################
## Author: Jaeseung Lee                                         ##
## Date: July 2024                                              ##
## Affiliation: POSTECH CSDL, South Korea                       ##
##################################################################

import subprocess as sp
import os
import time
from argparse import ArgumentParser

import utils


## SPICE Parameters of OTV PARFILE HSPICE
SPICE_parameters = {
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

## Write SPICE code for the Pixel3T1C circuit from the given parameters
def writePixelCircuit(file, SPICE_parameters, configs):
    number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio = configs
    with open(os.path.join(pixel_dir, sp_name), 'w') as file:
        # Write the header section
        file.write("*************************************************************************\n")
        file.write("***                                                                   ***\n")
        file.write("***                               < OTV >                             ***\n")
        file.write("***                                                                   ***\n")
        file.write("*************************************************************************\n")
        file.write("***************************   <Include File>   **************************\n")
        file.write(".INC '../OTV_PARFILE_Hspice_modify2'")

        file.write("\n.INC '../schematic.sp'\n")
        file.write("*************************************************************************\n")
        file.write(".savebias ./Initial.par tran time = 280u  ALL\n\n")
        file.write("**********   <Load Parameter>   ******************************************\n")
        # Write the parameter variations
        for param, value in SPICE_parameters.items():
            file.write('.PARAM {} = {}\n'.format(param, value))

        file.write("\n")
        file.write("*************************************************************************\n")
        file.write("***                                                                   ***\n")
        file.write("***                               <SPICE Param>                       ***\n")
        file.write("***                                                                   ***\n")
        file.write("*************************************************************************\n")

        file.write(f"\n*** Scan Signal Timing\n")
        file.write(f".PARAM T00 = 'TH*8 - TW_S - TMG'\n")
        file.write(f".PARAM T01 = 'T00 + TH'\n")
        for y in range(1,number_dataline_pixel):
            file.write(f".PARAM T0{y+1} = 'T0{y} + TH'\n")

        file.write(f"\n*** Scan Left Signal\n")
        for y in range(1,number_dataline_pixel+1):
            file.write(f"VSCAN_L<{y}> SCAN_L<{y}> 0 PULSE(GVGL VGH T0{y} TR TF TW_S '1/FR')\n")

        file.write(f"\n*** Scan Right Signal\n")
        for y in range(1,number_dataline_pixel+1):
            file.write(f"VSCAN_R<{y}> SCAN_R<{y}> 0 PULSE(GVGL VGH T0{y} TR TF TW_S '1/FR')\n")
        
        file.write('\n')
        # Write the header section
        file.write("\n")
        file.write("*************************************************************************\n")
        file.write("***                                                                   ***\n")
        file.write("***                               <Schematic>                         ***\n")
        file.write("***                                                                   ***\n")
        file.write("*************************************************************************\n")

        for x in range(1,number_scanline_pixel+1):
            for y in range(1,number_dataline_pixel+1):
                file.write(f"X_{x}_{y} DATAR{x}<{y+1}> DATAW{x}<{y+1}> DATAB{x}<{y+1}> DATAG{x}<{y+1}> DATAR{x}<{y}> DATAW{x}<{y}> DATAB{x}<{y}> DATAG{x}<{y}> SCAN{x}<{y}> SCAN{x+1}<{y}> VDD VDD VREF{x}<{y+1}> VREF{x}<{y}> VSS unit_pxl_ver01\n")        

        for y in range(1,number_dataline_pixel+1):
            file.write(f"X2<{y}> SCAN_R<{y}> SCAN{number_scanline_pixel+1}<{y}> scan_load_scale_top\n")
            file.write(f"X3<{y}> SCAN_L<{y}> SCAN1<{y}> scan_load_scale_top\n")

        for x in range(1,number_scanline_pixel+1):
            file.write(f"Xr{x} DATA_RU{x} DATAR{x}<1> data_load_scale_top\n")
            file.write(f"Xw{x} DATA_WU{x} DATAW{x}<1> data_load_scale_top\n")
            file.write(f"Xb{x} DATA_BU{x} DATAB{x}<1> data_load_scale_top\n")
            file.write(f"Xg{x} DATA_GU{x} DATAG{x}<1> data_load_scale_top\n")
            file.write('\n')

        for x in range(1,number_scanline_pixel+1):
            file.write(f"Xrp{x} DATA_R{x} DATA_RU{x} data_load_pcb_top\n")
            file.write(f"Xwp{x} DATA_W{x} DATA_WU{x} data_load_pcb_top\n")
            file.write(f"Xbp{x} DATA_B{x} DATA_BU{x} data_load_pcb_top\n")
            file.write(f"Xgp{x} DATA_G{x} DATA_GU{x} data_load_pcb_top\n")
            file.write('\n')

        for x in range(1,number_scanline_pixel+1):
            file.write(f"X6{x} VREF{x} VREF{x}<1> ref_load_scale_top\n")
        file.write(f"\n*** Vref Signal\n")
        for x in range(1,number_scanline_pixel+1):
            file.write(f"VREF{x} VREF{x} 0 DC VREF\n")
        
        file.write(f"\n*** Data Signal\n")
        for x in range(1,number_scanline_pixel+1):
            file.write(f"VDATA_W{x} DATA_W{x} GND PULSE(VDLW VDHW TB TR TF TW TP)\n")
            file.write(f"VDATA_R{x} DATA_R{x} GND PULSE(VDLR VDLR TB TR TF TW TP)\n")
            file.write(f"VDATA_G{x} DATA_G{x} GND PULSE(VDLG VDLG TB TR TF TW TP)\n")
            file.write(f"VDATA_B{x} DATA_B{x} GND PULSE(VDLB VDLB TB TR TF TW TP)\n")
            file.write("\n")

        file.write("\n")
        file.write("*************************************************************************\n")
        file.write("***                                                                   ***\n")
        file.write("***                                  <Run>                            ***\n")
        file.write("***                                                                   ***\n")
        file.write("*************************************************************************\n")
        file.write("\n")
        file.write(".PARAM  Px = 1920\n")
        file.write(".PARAM  Py = 100\n")
        file.write(f'.PARAM VDHW = {VDH}\n')
        file.write(f'.PARAM VDHR = {VDH}\n')
        file.write(f'.PARAM VDHG = {VDH}\n')
        file.write(f'.PARAM VDHB = {VDH}\n')
        file.write(f'.PARAM	loadratio = {load_ratio}\n')
        file.write("\n\n")
        file.write(".TRAN	20n '1/FR/2'\n")

        file.write("******************** Print node *****************************************\n")
        ## WHITE
        for x in range(1,number_scanline_pixel+1):
            for y in range(1,number_dataline_pixel+1):
                if (x % step_x == 1) and (y % step_y == 0):
                    file.write(f".PRINT v(dataw{x}<{y}>)\n")
                    file.write(f".PRINT V(X_{x}_{y}.XWHITE.drg)\n")
                    file.write(f".PRINT V(X_{x}_{y}.XWHITE.drs)\n")
                    file.write(f".PRINT I(X_{x}_{y}.XWHITE.XEL.d1)\n")

            if (x % step_x == 1):
                file.write(f".PRINT v(dataw{x}<2>)\n")
                file.write(f".PRINT V(X_{x}_2.XWHITE.drg)\n")
                file.write(f".PRINT V(X_{x}_2.XWHITE.drs)\n")
                file.write(f".PRINT I(X_{x}_2.XWHITE.XEL.d1)\n")

        for x in range(1,number_scanline_pixel+1):
            if x % step_x == 1:
                file.write(f".PRINT v(data_w{x})\n")
                file.write(f".PRINT v(data_wu{x})\n")
        
        for y in range(1,number_dataline_pixel+1):
            if y % step_y == 0:
                file.write(f".PRINT v(scan1<{y}>)\n")

        file.write(f".PRINT v(scan1<2>)\n")
        file.write("\n\n")
        file.write(".GLOBAL GND\n")
        file.write("*************************************************************************\n")
        file.write(".END\n")


if __name__ == '__main__':        
    # Open the SPICE netlist file for writing
    pixel_dir = '/project/common/LGD/spice_data/circuit'
    data_dir = '/project/common/LGD/spice_data/output'

    if not os.path.exists(pixel_dir):
        os.makedirs(pixel_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)    

    parser = ArgumentParser(description='SPICE')
    parser.add_argument('-s', '--number_scanline_pixel', required=True, type=int, default = '10', help='number')
    parser.add_argument('-d', '--number_dataline_pixel', required=True, type=int, default = '10', help='number')
    parser.add_argument('-r', '--res', required=False, type=float, default = '8.0', help='number')
    parser.add_argument('-c', '--cap', required=False, type=float, default = '0.008', help='number')
    parser.add_argument('-t', '--tw_s', required=False, type=int, default = '2', help='number')
    parser.add_argument('-v', '--VDH', required=False, type=float, default = '5.8', help='number')
    parser.add_argument('-lr', '--load_ratio', required=False, type=int, default = '2', help='number')
    parser.add_argument('-x', '--step_x', required=False, type=int, default = '10', help='number')
    parser.add_argument('-y', '--step_y', required=False, type=int, default = '8', help='number')
    args = parser.parse_args()

    # Argument Handling
    number_scanline_pixel = args.number_scanline_pixel
    number_dataline_pixel = args.number_dataline_pixel
    res = args.res
    cap = args.cap
    tw_s = 'TH*2'
    if args.tw_s == 1:
        args.tw_s = 'TH*1'
    VDH = args.VDH
    step_x = args.step_x
    step_y = args.step_y
    load_ratio = args.load_ratio
    configs = number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio

    # Generate the circuit name
    name = utils.getCircuitName('Pixel3T1C', configs)
    sp_name = f'{name}.sp'
    lis_name = f'{name}.lis'

    if os.path.exists(os.path.join(data_dir, sp_name)):
        print('Already Simulated')
        exit()        

    # Write the SPICE code for the Pixel3T1C circuit
    writePixelCircuit(sp_name, SPICE_parameters, configs)

    # Run the HSPICE simulation (can be alternated with Other SPICE simulators)
    start = time.time()
    sp.call(f'hspice -i {os.path.join(pixel_dir, sp_name)} -o {os.path.join(data_dir, lis_name)}', shell=True)
    print(f"{name} : TIME {time.time() - start:.2f}")
    