# PyMOL script to visualize LRP1 binding epitope on MDK
    
# Load structure
load data/structures/1mkc.pdb, MDK

# Set initial view
hide everything
show cartoon, MDK
color grey80, MDK

# Highlight epitope residues
select lrp1_epitope, MDK and resi 62+63+64+66+69+72+78+79+80+81+82+83+84+85+86+87+88+89+90+91+92+94+96+97+98+99+100+101+102+103+104
show sticks, lrp1_epitope
color red, lrp1_epitope

# Label key residues
select key_basic, MDK and resn ARG+LYS and resi 62+63+64+66+69
label key_basic and name CA, "%s-%s" % (resn, resi)

# Create surface representation
show surface, MDK
set transparency, 0.5, MDK

# Color surface by epitope
color salmon, lrp1_epitope

# Set view
zoom MDK
orient

# Ray trace for publication quality
bg_color white
set ray_shadows, 0
# ray 1200, 1200

# Save session
save results/visualizations/lrp1_epitope.pse
