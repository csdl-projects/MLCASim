import os

if __name__ == "__main__":

    # 80x45_try 작성

    hh = 80
    vv = 45
    step_x = 10 # Measure Point
    step_y = 8

    file_path = os.path.join(os.getcwd(), f'{hh}x{vv}_try5')

    lines = []
    lines.append(f"\n*** Pixel Array {hh}x{vv} Netlist\n")
    for x in range(1,hh+1):
        for y in range(1,vv+1):
            lines.append(f"X_{x}_{y} DATAR{x}<{y+1}> DATAW{x}<{y+1}> DATAB{x}<{y+1}> DATAG{x}<{y+1}> DATAR{x}<{y}> DATAW{x}<{y}> DATAB{x}<{y}> DATAG{x}<{y}> SCAN{x}<{y}> SCAN{x+1}<{y}> VDD VDD VREF{x}<{y+1}> VREF{x}<{y}> VSS unit_pxl_ver01\n")

    lines.append(f"\n*** Gate IC Netlist\n")
    for y in range(1,vv+1):
        lines.append(f"X2<{y}> SCAN_R<{y}> SCAN{hh+1}<{y}> scan_load_scale_top\n")
        lines.append(f"X3<{y}> SCAN_L<{y}> SCAN1<{y}> scan_load_scale_top\n")

    lines.append(f"\n*** Data IC Netlist\n")
    for x in range(1,hh+1):
        lines.append(f"Xr{x} DATA_RU{x} DATAR{x}<1> data_load_scale_top\n")
        lines.append(f"Xw{x} DATA_WU{x} DATAW{x}<1> data_load_scale_top\n")
        lines.append(f"Xb{x} DATA_BU{x} DATAB{x}<1> data_load_scale_top\n")
        lines.append(f"Xg{x} DATA_GU{x} DATAG{x}<1> data_load_scale_top\n")
        lines.append('\n')

    for x in range(1,hh+1):
        lines.append(f"Xrp{x} DATA_R{x} DATA_RU{x} data_load_pcb_top\n")
        lines.append(f"Xwp{x} DATA_W{x} DATA_WU{x} data_load_pcb_top\n")
        lines.append(f"Xbp{x} DATA_B{x} DATA_BU{x} data_load_pcb_top\n")
        lines.append(f"Xgp{x} DATA_G{x} DATA_GU{x} data_load_pcb_top\n")
        lines.append('\n')

    for x in range(1,hh+1):
        lines.append(f"X6{x} VREF{x} VREF{x}<1> ref_load_scale_top\n")

    lines.append(f"\n*** Vref Signal\n")
    for x in range(1,hh+1):
        lines. append(f"VREF{x} VREF{x} 0 DC VREF\n")
    
    lines.append(f"\n*** Data Signal\n")
    for x in range(1,hh+1):
        lines.append(f"VDATA_W{x} DATA_W{x} GND PULSE(VDLW VDHW TB TR TF TW TP)\n")
        lines.append(f"VDATA_R{x} DATA_R{x} GND PULSE(VDLR VDLR TB TR TF TW TP)\n")
        lines.append(f"VDATA_G{x} DATA_G{x} GND PULSE(VDLG VDLG TB TR TF TW TP)\n")
        lines.append(f"VDATA_B{x} DATA_B{x} GND PULSE(VDLB VDLB TB TR TF TW TP)\n")
        lines.append("\n")

    lines.append(f"\n*** Scan Signal Timing\n")
    lines.append(f".PARAM T00 = 'TH*8 - TW_S - TMG'\n")
    lines.append(f".PARAM T01 = 'T00 + TH'\n")
    for y in range(1,vv):
        lines.append(f".PARAM T0{y+1} = 'T0{y} + TH'\n")

    lines.append(f"\n*** Scan Left Signal\n")
    for y in range(1,vv+1):
        lines.append(f"VSCAN_L<{y}> SCAN_L<{y}> 0 PULSE(GVGL VGH T0{y} TR TF TW_S '1/FR')\n")

    lines.append(f"\n*** Scan Right Signal\n")
    for y in range(1,vv+1):
        lines.append(f"VSCAN_R<{y}> SCAN_R<{y}> 0 PULSE(GVGL VGH T0{y} TR TF TW_S '1/FR')\n")
    
    lines.append(f"\n*** Measure\n")
    for x in range(1,hh+1):
        for y in range(1,vv+1):

            if (x % step_x == 1) and (y % step_y == 0):
                lines.append(f".PROBE v(dataw{x}<{y}>)\n")
                lines.append(f".PROBE V(X_{x}_{y}.XWHITE.drg)\n")
                lines.append(f".PROBE V(X_{x}_{y}.XWHITE.drs)\n")
                lines.append(f".PROBE I(X_{x}_{y}.XWHITE.XEL.d1)\n")
        if (x % step_x == 1):
            lines.append(f".PROBE v(dataw{x}<2>)\n")
            lines.append(f".PROBE V(X_{x}_2.XWHITE.drg)\n")
            lines.append(f".PROBE V(X_{x}_2.XWHITE.drs)\n")
            lines.append(f".PROBE I(X_{x}_2.XWHITE.XEL.d1)\n")

    for x in range(1,hh+1):
        if x % step_x == 1:
            lines.append(f".PROBE v(data_w{x})\n")
            lines.append(f".PROBE v(data_wu{x})\n")
    
    for y in range(1,vv+1):
        if y % step_y == 0:
            lines.append(f".PROBE v(scan1<{y}>)\n")
    lines.append(f".PROBE v(scan1<2>)\n")

    f = open(file_path, 'w')
    for line in lines:
        f.write(line)
    f.close()

    print("Done!")
