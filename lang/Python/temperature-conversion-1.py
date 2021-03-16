while True:
    k = float(input('K ? '))
    print("%g Kelvin = %g Celsius = %g Fahrenheit = %g Rankine degrees."
          % (k, k - 273.15, k * 1.8 - 459.67, k * 1.8))
