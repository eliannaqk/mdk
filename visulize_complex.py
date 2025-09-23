#!/usr/bin/env python3
"""
PyMOL visualization script for MDK-nanobody complexes
Run with: pymol -r visualize_complex.py
"""

import pymol
from pymol import cmd, stored
import json
from pathlib import Path
import numpy as np

def load_structures():
    """Load MDK and nanobody structures"""
    
    # Load MDK C-terminal domain
    mdk_path = "data/structures/1mkc.pdb"
    if Path(mdk_path).exists():
        cmd.load(mdk_path, "MDK")
        print(f"Loaded MDK structure: {mdk_path}")
    
    # Load epitope definition
    epitope_file = Path("data/epitopes/lrp1_patch.json")
    if epitope_file.exists():
        with open(epitope_file, 'r') as f:
            epitope_data = json.load(f)
        return epitope_data
    
    return None

def visualize_lrp1_epitope(epitope_data):
    """Highlight the LRP1 binding epitope on MDK"""
    
    if not epitope_data:
        print("No epitope data found")
        return
    
    # Get epitope residues
    core_residues = epitope_data.get('core_residues', [])
    expanded_residues = epitope_data.get('expanded_residues', [])
    
    # Create selections
    if core_residues:
        core_selection = f"MDK and resi {'+'.join(map(str, core_residues))}"
        cmd.select("core_epitope", core_selection)
    
    if expanded_residues:
        expanded_selection = f"MDK and resi {'+'.join(map(str, expanded_residues))}"
        cmd.select("lrp1_epitope", expanded_selection)
    
    # Color scheme
    cmd.color("grey80", "MDK")
    cmd.color("salmon", "lrp1_epitope")
    cmd.color("red", "core_epitope")
    
    # Show representations
    cmd.show("cartoon", "MDK")
    cmd.show("sticks", "core_epitope")
    cmd.show("surface", "lrp1_epitope")
    cmd.set("transparency", 0.3, "lrp1_epitope")
    
    # Label key residues
    for res in core_residues[:5]:  # Label first 5 core residues
        cmd.label(f"MDK and resi {res} and name CA", f'"{res}"')
    
    print(f"Visualized LRP1 epitope: {len(core_residues)} core, {len(expanded_residues)} total residues")

def create_binding_site_mesh():
    """Create mesh representation of binding site"""
    
    # Create mesh for epitope surface
    cmd.create("epitope_mesh", "lrp1_epitope")
    cmd.hide("everything", "epitope_mesh")
    cmd.show("mesh", "epitope_mesh")
    cmd.set("mesh_color", "orange", "epitope_mesh")
    cmd.set("mesh_width", 0.5)
    
    print("Created binding site mesh")

def analyze_electrostatics():
    """Analyze and visualize electrostatic properties"""
    
    # Color by charge
    cmd.select("positive", "resn ARG+LYS+HIS")
    cmd.select("negative", "resn ASP+GLU")
    cmd.select("hydrophobic", "resn ALA+VAL+LEU+ILE+MET+PHE+TRP+PRO")
    
    # Create new object for electrostatics
    cmd.create("MDK_electro", "MDK")
    cmd.hide("everything", "MDK_electro")
    cmd.show("surface", "MDK_electro")
    
    # Color by properties
    cmd.color("blue", "MDK_electro and positive")
    cmd.color("red", "MDK_electro and negative")
    cmd.color("white", "MDK_electro and hydrophobic")
    
    cmd.set("transparency", 0.5, "MDK_electro")
    
    print("Created electrostatic surface")

def setup_nanobody_docking_view():
    """Setup view for nanobody docking visualization"""
    
    # Create coordinate axes for reference
    axes = [
        ["HETATM    1  X   AXS     1       0.000   0.000   0.000  1.00  0.00           X", "red"],
        ["HETATM    2  X   AXS     1      10.000   0.000   0.000  1.00  0.00           X", "red"],
        ["HETATM    3  Y   AXS     1       0.000   0.000   0.000  1.00  0.00           Y", "green"],
        ["HETATM    4  Y   AXS     1       0.000  10.000   0.000  1.00  0.00           Y", "green"],
        ["HETATM    5  Z   AXS     1       0.000   0.000   0.000  1.00  0.00           Z", "blue"],
        ["HETATM    6  Z   AXS     1       0.000   0.000  10.000  1.00  0.00           Z", "blue"]
    ]
    
    # Add axes as pseudoatoms
    for i in range(0, len(axes), 2):
        obj_name = f"axis_{axes[i][1]}"
        cmd.pseudoatom(obj_name, pos=[0, 0, 0])
        cmd.color(axes[i][1], obj_name)
    
    # Create potential nanobody positions
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    radius = 25.0
    
    for i, angle in enumerate(angles):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        
        cmd.pseudoatom(f"nb_position_{i+1}", pos=[x, y, z])
        cmd.show("sphere", f"nb_position_{i+1}")
        cmd.set("sphere_scale", 2.0, f"nb_position_{i+1}")
        cmd.color("cyan", f"nb_position_{i+1}")
    
    print("Setup nanobody docking positions")

def create_publication_figures():
    """Generate publication-quality figures"""
    
    # Figure 1: Overall structure with epitope
    cmd.set_view([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -50.0,
        0.0, 0.0, 0.0,
        40.0, 100.0, -20.0
    ])
    
    cmd.bg_color("white")
    cmd.set("ray_shadows", 0)
    cmd.set("ambient", 0.3)
    cmd.set("direct", 0.7)
    
    cmd.ray(1200, 1200)
    cmd.png("results/visualizations/mdk_epitope_overview.png", dpi=300)
    
    # Figure 2: Close-up of epitope
    cmd.zoom("lrp1_epitope", buffer=5)
    cmd.ray(1200, 1200)
    cmd.png("results/visualizations/lrp1_epitope_closeup.png", dpi=300)
    
    # Figure 3: Electrostatic surface
    cmd.disable("all")
    cmd.enable("MDK_electro")
    cmd.zoom("MDK_electro")
    cmd.ray(1200, 1200)
    cmd.png("results/visualizations/mdk_electrostatics.png", dpi=300)
    
    print("Generated publication figures")

def create_movie():
    """Create a rotating movie of the complex"""
    
    # Setup movie
    cmd.mset("1 x360")
    cmd.util.mroll(1, 360, 1)
    
    # Set quality
    cmd.set("ray_trace_frames", 1)
    cmd.set("cache_frames", 0)
    
    # Movie settings
    cmd.viewport(800, 600)
    cmd.set("ray_shadows", 0)
    
    # Save movie frames
    cmd.mpng("results/visualizations/movie/frame", first=1, last=360)
    
    print("Created movie frames in results/visualizations/movie/")
    print("Convert to video with: ffmpeg -r 30 -i frame%04d.png -c:v libx264 mdk_rotation.mp4")

def save_session():
    """Save PyMOL session"""
    
    session_file = "results/visualizations/mdk_lrp1_epitope.pse"
    cmd.save(session_file)
    print(f"Saved PyMOL session: {session_file}")

def main():
    """Main visualization pipeline"""
    
    print("=" * 60)
    print("MDK-LRP1 Epitope Visualization")
    print("=" * 60)
    
    # Initialize PyMOL
    pymol.finish_launching(['pymol', '-q'])
    
    # Load structures
    print("\n1. Loading structures...")
    epitope_data = load_structures()
    
    # Visualize epitope
    print("\n2. Visualizing LRP1 epitope...")
    visualize_lrp1_epitope(epitope_data)
    
    # Create mesh
    print("\n3. Creating binding site mesh...")
    create_binding_site_mesh()
    
    # Analyze electrostatics
    print("\n4. Analyzing electrostatics...")
    analyze_electrostatics()
    
    # Setup docking view
    print("\n5. Setting up docking view...")
    setup_nanobody_docking_view()
    
    # Generate figures
    print("\n6. Generating publication figures...")
    Path("results/visualizations").mkdir(parents=True, exist_ok=True)
    create_publication_figures()
    
    # Save session
    print("\n7. Saving session...")
    save_session()
    
    # Final view
    cmd.zoom("MDK")
    cmd.orient()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("\nPyMOL commands:")
    print("  - 'zoom lrp1_epitope' to focus on epitope")
    print("  - 'enable MDK_electro' to show electrostatics")
    print("  - 'rock' to start rocking motion")
    print("  - 'mclear' to clear movie")

if __name__ == "__main__":
    main()
    
    # Keep PyMOL open
    pymol.cmd.extend("visualize_epitope", visualize_lrp1_epitope)
    pymol.cmd.extend("analyze_electro", analyze_electrostatics)