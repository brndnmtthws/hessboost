# Renders the docs/performance.md XGBoost charts from xgboost.dat.
#
#     gnuplot -c docs/benchmarks/charts.gp
#
# Run from the repository root. The .dat values mirror the XGBoost
# comparison table in docs/performance.md. After refreshing the table and
# the .dat file, regenerate the SVGs with the command above. Requires
# gnuplot 5.4 or newer.
#
# Cluster offsets: with `clustered gap 1`, a cluster of N bars spans
# N/(N+1) x units, so bar k of N sits at (2k - N + 1) / (2(N+1)).

set datafile commentschars "#"
xgb_data = "docs/benchmarks/xgboost.dat"

# hessboost is green and XGBoost blue; darker shades are lower
# thread counts throughout.
seq_t1 = "#1b5e20"
seq_t4 = "#43a047"
seq_t16 = "#a5d6a7"
xgb_t1 = "#1565c0"
xgb_t4 = "#1e88e5"
xgb_t16 = "#90caf9"

set style fill solid 1.0 border lt -1
set style histogram clustered gap 1
set style data histograms
set boxwidth 0.9
set grid ytics lc rgb "#d0d0d0" lw 1
set border 3
set tics nomirror out
set xtics scale 0
set bmargin 5

# SVG output is resolution independent; the size only sets the aspect
# ratio and the font-to-plot proportions.
set terminal svg size 960,520 dynamic font "sans-serif,13" background rgb "white"

# --- Speedup over XGBoost by thread count --------------------------------------

set output "docs/benchmarks/xgboost-speedup.svg"

set title "hessboost speedup over XGBoost 3.4.1 (higher is better)\n{/*0.8 median of six fits, Apple M3 Max, CPU hist, 100 rounds}"
set ylabel "fit time ratio, XGBoost / hessboost"
set yrange [0:5.5]
set ytics 1.0
set arrow 1 from -0.5,1 to 3.5,1 nohead lc rgb "#303030" dt 2 lw 1.5 front
set key top right reverse Left samplen 2 spacing 1.3
plot xgb_data using ($3/$2):xtic(1) lc rgb seq_t1  title "1 thread", \
     xgb_data using ($5/$4)          lc rgb seq_t4  title "4 threads", \
     xgb_data using ($7/$6)          lc rgb seq_t16 title "16 threads", \
     xgb_data using ($0-0.25):($3/$2):(sprintf("%.2fx", $3/$2)) \
          with labels offset 0,0.6 font ",9" notitle, \
     xgb_data using ($0+0.00):($5/$4):(sprintf("%.2fx", $5/$4)) \
          with labels offset 0,0.6 font ",9" notitle, \
     xgb_data using ($0+0.25):($7/$6):(sprintf("%.2fx", $7/$6)) \
          with labels offset 0,0.6 font ",9" notitle, \
     keyentry with lines lc rgb "#303030" dt 2 lw 1.5 title "XGBoost = 1.0x"

unset arrow 1

# --- Median fit time by workload and thread count ------------------------------

set output "docs/benchmarks/xgboost-threads.svg"

set title "Median fit time, hessboost vs XGBoost 3.4.1 (lower is better)\n{/*0.8 seconds, log scale, fresh matrix construction plus training}"
set ylabel "fit time (s)"
set logscale y 10
set yrange [0.15:5]
set ytics ("0.2" 0.2, "0.5" 0.5, "1" 1, "2" 2, "5" 5)
set mytics 10
set key top left reverse Left samplen 1.5 spacing 1.1

plot xgb_data using 2:xtic(1) lc rgb seq_t1  title "hessboost, 1 thread", \
     xgb_data using 3:xtic(1) lc rgb xgb_t1  title "XGBoost, 1 thread", \
     xgb_data using 4:xtic(1) lc rgb seq_t4  title "hessboost, 4 threads", \
     xgb_data using 5:xtic(1) lc rgb xgb_t4  title "XGBoost, 4 threads", \
     xgb_data using 6:xtic(1) lc rgb seq_t16 title "hessboost, 16 threads", \
     xgb_data using 7:xtic(1) lc rgb xgb_t16 title "XGBoost, 16 threads"

unset logscale
