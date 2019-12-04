import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)

pgm = daft.PGM()
pgm.add_node("Temperature", r"Temperature", 2, 2, aspect=2.7)
pgm.add_node("Hunger", r"Hunger Level", 4, 2, aspect=3.1)
pgm.add_node("Ice", r"Ice Cream", 3, 1, aspect=2.4, observed=True)
pgm.add_edge("Temperature", "Ice")
pgm.add_edge("Hunger", "Ice")

pgm.render()
pgm.savefig("bat.png", dpi=150)


pgm = daft.PGM()
pgm.add_node("Size", r"Cell Size Uniformity", 0.5, 1, aspect=4.5)
pgm.add_node("Shape", r"Cell Shape Uniformity", 0.5, 2, aspect=4.5)
pgm.add_node("chromatin", r"Bland Chromatin", 3, 2, aspect=3.5)
pgm.add_node("nucleoli", r"Normal Nucleoli", 5, 2, aspect=3.3)
pgm.add_node("nuclei", r"Bare Nuclei", 6, 1, aspect=2.6)
pgm.add_node("class", r"Class", 4, 1, aspect=2, observed=True)
pgm.add_edge("Shape", "Size")
pgm.add_edge("chromatin", "class")
pgm.add_edge("nucleoli", "class")
pgm.add_edge("nucleoli", "nuclei")

pgm.render()
pgm.savefig("cb_mle.png", dpi=150)