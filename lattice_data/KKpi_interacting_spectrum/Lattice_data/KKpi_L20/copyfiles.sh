#!/bin/bash

#ls /work/JLabLQCD/LHPC/Spectrum/Clover/NF2+1/szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per/redstar/KKpi.S2I2/fits_mhi/000_A1m/t011/MassJackFiles/mass_t0_11_reorder_state*
t0val="12"
for i in `ls /work/JLabLQCD/LHPC/Spectrum/Clover/NF2+1/szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per/redstar/KKpi.S2I2/fits_mhi/000_A1m/t011/MassJackFiles/mass_t0_11_reorder_state*`; 
do a=`echo $i | sed "s|/|_|g" | sed "s|_work_JLabLQCD_LHPC_Spectrum_Clover_NF2+1_szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per_redstar_KKpi.S2I2_fits_mhi|mass|g"`;
cp $i ./$a;
#echo $a;
done

for i in `ls /work/JLabLQCD/LHPC/Spectrum/Clover/NF2+1/szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per/redstar/KKpi.S2I2/fits_mhi/100_A2/t012/MassJackFiles/mass_t0_12_reorder_state*`;
do a=`echo $i | sed "s|/|_|g" | sed "s|_work_JLabLQCD_LHPC_Spectrum_Clover_NF2+1_szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per_redstar_KKpi.S2I2_fits_mhi|mass|g"`;
cp $i ./$a;
#echo $a;
done


for i in `ls /work/JLabLQCD/LHPC/Spectrum/Clover/NF2+1/szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per/redstar/KKpi.S2I2/fits_mhi/110_A2/t013/MassJackFiles/mass_t0_13_reorder_state*`;
do a=`echo $i | sed "s|/|_|g" | sed "s|_work_JLabLQCD_LHPC_Spectrum_Clover_NF2+1_szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per_redstar_KKpi.S2I2_fits_mhi|mass|g"`;
cp $i ./$a;
#echo $a;
done

for i in `ls /work/JLabLQCD/LHPC/Spectrum/Clover/NF2+1/szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per/redstar/KKpi.S2I2/fits_mhi/111_A2/t012/MassJackFiles/mass_t0_12_reorder_state*`;
do a=`echo $i | sed "s|/|_|g" | sed "s|_work_JLabLQCD_LHPC_Spectrum_Clover_NF2+1_szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per_redstar_KKpi.S2I2_fits_mhi|mass|g"`;
cp $i ./$a;
#echo $a;
done

for i in `ls /work/JLabLQCD/LHPC/Spectrum/Clover/NF2+1/szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per/redstar/KKpi.S2I2/fits_mhi/200_A2/t010/MassJackFiles/mass_t0_10_reorder_state*`;
do a=`echo $i | sed "s|/|_|g" | sed "s|_work_JLabLQCD_LHPC_Spectrum_Clover_NF2+1_szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265_per_redstar_KKpi.S2I2_fits_mhi|mass|g"`;
cp $i ./$a;
#echo $a;
done
