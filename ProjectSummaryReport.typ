#set page(
  paper: "us-letter",
  numbering: "1", 
  number-align: top + right,
)
#set par(
  leading: 1.5em,
  spacing: 2em,
  first-line-indent: (
    amount: 1.5em,
    all: true
  )
)
#set text(size: 12pt)

#show figure.where(kind: table): set figure.caption(position: top)

#align(center, text(17pt)[
  *Tubular Flow Reactor Project Summary Report*
])

The goal of the experiment was to determine the effect of flow rate on the
conversion of the reaction between sodium hydroxide and ethyl acetate. For this
experiment, the flow rates 0.25 GPM, 0.45 GPM, and 1 GPM were tested to observe
the effect different flow regimes have on conversion. As part of this
experiment, residence time distribution models (RTDs) were used to compare the
results obtained from experimentation.

The tubular flow reactor consists of, two reagent storage tanks, two equal
sized reagent holding tanks, one neutralization tank, two conductivity meters
placed on the entrance and exit of the tubular reactor, two pumps for mixing
the reagents, one pump to push fluid through the reactor, three control valves
and flow meters to monitor flow of reagents into the reactor. To preform the
experiment, solutions of 0.08 M and 0.1 M of sodium hydroxide and ethyl acetate
were made, respectively. A sample of pure sodium hydroxide is drawn off to be
titrated and its initial concentration calculated. Both reagents are fed into
the reactor at equal flow rates. Each flow rate was reacted at 175% of its
residence time to ensure steady state. After each flow rate was tested, the
holding tanks were closed off and the reagents in the reactor were left to
react until completion, after which the system was flushed.

By using conductivity data and the initial concentration of sodium hydroxide,
the concentration of sodium hydroxide can be calculated and plugged into the
equation below to determine the concentration.
$ X = 1 - C_("NaOH") / C_("NaOH,0") $
To ensure that the reported conversion only accounts for conversion of the flow
rate in question, the average is taken after one residence time, the average
time a particle spends in a reactor. Due to time constraints and cost of
materials, only one trial could be ran, so the error reported on the
experimental conversions is due to systematic error.

For the theoretical conversion, the flow regime dictated the
model used. For turbulent flow, an ideal plug flow reactor and RTD model were
used. For laminar flow, a laminar flow reactor and RTD model were used. Due to
the unpredictability of transition flow, no model exist specifically for
transition. So, using the laminar flow and plug flow reactor models, a range of
conversions was determined. A complete summary of data collected is presented
in @results.

#figure(
  caption: [Summary of experimentally obtained conversion compared to the conversion
calculated by the appropriate Residence Time Distribution Model],
  kind: table,
  table(
    columns: 4,
    align: horizon,
    table.cell(rowspan: 2)[Flow Regime (Flow rate, gpm)],table.cell(colspan: 2)[Conversion], table.cell(rowspan: 2)[Theoretical Model],
    [Experimental], [Theoretical],

    [Laminar (0.25)],
    [$0.491 plus.minus 0.00669$],[$0.449$], [Laminar #footnote[Estimated error of $4.841 dot 10^(-10)$.]],

    table.cell(rowspan: 2)[Transition (0.45)],
    table.cell(rowspan: 2)[$0.382 plus.minus 0.00661$], [$0.311$], [Laminar #footnote[Estimated error of $5.937 dot 10^(-11)$.]],
    [$0.347$], [Turbulent],

    [Turbulent (1.00)],
    [$0.221 plus.minus 0.00823$], [$0.188$], [Turbulent]
  )
)<results>

To answer the original objective posed at the start of this experiment, as flow
rate through a tubular reactor increases, the conversion of a reaction
decreases. This trend matches the trend of the theoretical models used. Laminar
flow will yield higher conversion than that of turbulent flow. So, depending on
the application a tubular reactor is used in, if a high conversion is needed,
small flow rates must be used. If small flow rates are not achievable, a
recycle stream may need to used and unused reagents fed back into the reactor.

#pagebreak()

= Appendix

#show math.equation.where(block: true): set block(spacing: 1em)

== Sample Calculations
=== Experimental

Calculate the internal liquid volume of the cylindrical reactor. Use the
measured length l and radius r and substitute numeric values to give the
reactor volume in cubic inches, which is then converted to gallons for
comparison with flow rates. This volume is needed to compute residence time and
to relate liquid volume to flow.

$ V_r = pi dot l dot r^2 $
$ V_r = pi dot (952 "in") dot (0.25 "in")^2 $
$ V_r = 186.9 "in"^3 = 0.809 "gal" $

Compute the mean residence time (average time a fluid element spends in the
reactor) by dividing the reactor volume $V_r$ by the volumetric flow rate
$dot(V)$ (gallons per minute) that is being tested for that trial. The residence
time is used to determine the appropriate reaction times and to compare them
with experimental sampling times, so all relevant data is collected for the
trial. 

$ tau = V_r / dot(V) $
$ tau = (0.809 "gal") / (0.45 "GPM") $

Multiply the residence time by the factor 1.75 to get the target
sampling/processing time used in the experiment. For this case, to get the time
a reaction is considered effectively complete. This gives a time in seconds
consistent with the residence time units. 

$ tau = 1.80 "min" = 107.9 "sec" $
$ 1.75 dot tau = 188.8 "sec" $

The conductivity reading from the solution is used as the calibration signal K.
Species A is the limiting reactant, NaOH, while species B is ethyl acetate. To
convert the raw conductivity measurement into the actual concentration of
species A, the signal is normalized between the baseline conductivity
$K_infinity$ representing the fully reacted state, and the reference
conductivity $K_0$, representing the initial, unreacted state. This linear
calibration relationship allows conductivity values to be directly converted
into concentrations.
#let cond = $space (mu S) / ("cm")$
$ C_A = (K - K_infinity) / (K_0 - K_infinity) dot C_("A,0") $
$ C_A = ( 6915.138 cond - 6642.424 cond) / (7455.158 cond - 6642.424 cond) dot 0.0068167 space "mol" / "L"  $

Calculate the average fractional conversion $overline(X)$ of reactant A
time t using the concentration $C_A$ that is found above. This gives the
fraction of A that has reacted, where $overline(X) = 0$ means no conversion
and $overline(X)$ means complete conversion. This formula was used to
calculate conversion for each second the trial was running and averaged the
data points for each flow regime together. Below is an example of one of those
points for the transition flow regime.

#let mol = $space "mol"/"L"$
$ overline(X) = 1 - C_A / C_(A",0") $
$ overline(X) = 1 - (0.02287 mol) / (0.068167 mol) $
$ overline(X) = 0.664 $

The initial concentration of species A was calculated by making an aqueous
solution using the primary standard potassium hydrogen phthalate (KHP). The
amount of volume displaced in the burette was multiplied by the molarity of the
KHP solution, specifying the moles in the reactant A sample. The number of
moles can be divided by the volume of sample, resulting in the initial
concentration of reactant A. 

=== Theoretical

The rate constant is calculated using the Arrhenius equation, which relates the
frequency factor, activation energy, universal gas constant, and reaction
temperature. This determines the specific rate constant for the reaction
undergone in this unit. Both $K_0$ and $E_A$ were sourced at room temperature.

$ k = k_0 dot e^(-E_A/(R dot T)) $
$ k = 1.05 dot 10^3 "m"^3/("mol" dot "s") 
e^((-39.8 "kJ" / "mol" dot 1000 "J" / "kJ") / (8.314 "J"/("mol" dot "K") dot 295.49 "K")) $
$ k = 9.283 dot 10^(-5) "m"^3 / ("mol" dot "K") $

The overall conversion is expressed as an integral of the instantaneous
conversion weighted by the residence time distribution (RTD). This accounts for
the distribution of residence times in the reactor and provides the expected
overall conversion. The RTD is dependent on the flow regime the molecules are
in. 

$ overline(X) = integral_0 ^infinity X(t)E(t) dif t $
$ 
X(t) = (C_"B,0" dot (1 - e^(-k dot (C_"B,0" - C_"A,0") dot t))) /
(C_"B,0" - C_"A,0" dot e^(-k (C_"B,0" - C_"A,0")dot t))
$

==== Laminar Flow
For laminar flow, the residence time distribution is uniform between half the
mean residence time and infinity. Substitute this distribution into the general
conversion integral gives the theoretical conversion under laminar flow
conditions.

#set math.cases(gap: 1em)
#let Cao = $0.076854 / 2$
#let Cbo = $0.1 / 2$

$
E(t) = cases(
  0 ", " t < 0,
  tau^2/(2t^3) ", " t >= tau/ 2,
)
$

$
overline(X) = integral_(tau/2) ^(infinity)
tau^2/(2t^3) dot
(C_"B,0" dot (1 - e^(-k dot (C_"B,0" - C_"A,0") dot t))) /
(C_"B,0" - C_"A,0" dot e^(-k (C_"B,0" - C_"A,0")dot t))
dif t
$

#let kcon = $ -9.283 dot 10^(-5) "m"^3 / ("mol" dot "s") dot (1000 "L") / (1 "m"^3)$

$
overline(X) = integral_(tau/2) ^(infinity)
tau^2/(2t^3) dot

(Cbo "M" dot 1 - e^(kcon (Cbo "M" - Cao "M")dot t)) /
(Cbo "M" - Cao "M" dot e^(kcon (Cbo "M" - Cao "M") dot t))
dif t
$

$ overline(X) = 0.491 $

==== Turbulent Flow
For turbulent flow, all fluid elements are assumed to have the same residence
time equal to the mean residence time. Therefore, the overall conversion
reduces to evaluating the instantaneous conversion at that time only.
Additionally, transitional flow is also modeled using the equation below. 

$ overline(X) = integral_0^infinity X(t)E(t) dif t
= cases(
  0 "if" t eq.not tau,
  X(tau) "if" t eq tau,
)
$

$
overline(X) = 
(Cbo "M" dot (1 - e^(kcon (Cbo "M" - Cao "M") dot 48.55 "sec"))) /
(Cbo "M" -Cao "M" dot e^(kcon (Cbo "M" - Cao "M") dot 48.55 "sec"))
$

$
overline(X) =
0.221
$

=== Systematic Error

All conductivity readings were measured using the Model 54eC conductivity HART
analyzer. The sensor used in the analyzer is a contacting conductivity
sensor For contacting conductivity sensors, the analyzer is accurate
within ±0.5% of the reading as reflected in the calculations below 

$
Delta_("X,Tran") = sqrt(
  ((partial X) / (partial k) dot delta_k)^2 +
  ((partial X) / (partial k_infinity) dot delta_k_infinity)^2 +
  ((partial X) / (partial k_0) dot delta_k_0)^2
)
$

$
Delta_("X,Tran") = sqrt(
  (-1 / (k_0 - k_infinity) dot delta_k)^2 +
  (1 / (k_0 - k_infinity) - (k - k_infinity) / ((k_0 - k_infinity)^2) dot delta_k_infinity)^2 +
  ((k - k_infinity) / ((k_0 - k_infinity)^2) dot delta_k_0)^2
)
$

#let k0 = $7912.609 cond$
#let kinf = $2809.877 cond$
#let kt = $5318.821 cond$

$ Delta_("X,Tran") =
sqrt(
  (-1/ (7455.158 cond - 6642.424 cond) dot delta_k)^2 + \
  (-1 / (7455.158 cond - 6642.424 cond) - (6915.138 cond - 6642.424 cond) / (7455.158 cond - kinf)^2 dot delta_k_inf)^2 + \
  ((kt - kinf) / (k0 - kinf)^2 dot delta_k_inf)^2
)
$

$
Delta_("X,Tran") =
sqrt(
  (-0.0012304149 dot (0.005 *kt))^2 + \
  (0.000817547 dot (0.005 * kinf))^2 + \
  (0.0004128674 dot (0.005 * k0))^2
)
$

$
Delta_("X,Tran") = 0.006607
$
