# Spatial ResolVI Debugging Helper
# This script will help diagnose what's causing your initialization error

import os
import numpy as np
import scanpy as sc
import sys

def diagnose_spatial_data(adata_path):
    """
    Diagnose spatial data issues in your AnnData object
    """
    print("🔍 DIAGNOSING SPATIAL DATA ISSUES...")
    print("=" * 50)
    
    # Load data
    try:
        print(f"📂 Loading data from: {adata_path}")
        adata = sc.read(adata_path)
        print(f"✅ Data loaded successfully: {adata.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Check basic data structure
    print(f"\n📊 BASIC DATA INFO:")
    print(f"   Shape: {adata.shape}")
    print(f"   Obs columns: {list(adata.obs.columns)}")
    print(f"   Obsm keys: {list(adata.obsm.keys())}")
    
    # Check for spatial data
    print(f"\n🗺️  SPATIAL DATA CHECK:")
    
    spatial_candidates = []
    if 'spatial' in adata.obsm:
        spatial_candidates.append('spatial')
        print(f"   ✅ Found 'spatial' in obsm: {adata.obsm['spatial'].shape}")
    
    # Look for other spatial coordinate keys
    spatial_keys = [key for key in adata.obsm.keys() if 'spatial' in key.lower() or 'coord' in key.lower()]
    for key in spatial_keys:
        if key not in spatial_candidates:
            spatial_candidates.append(key)
            print(f"   ✅ Found '{key}' in obsm: {adata.obsm[key].shape}")
    
    if not spatial_candidates:
        print(f"   ❌ No spatial coordinates found!")
        print(f"   Available obsm keys: {list(adata.obsm.keys())}")
        return None
    
    # Check spatial data format
    for key in spatial_candidates:
        spatial_data = adata.obsm[key]
        print(f"\n   📍 Analyzing '{key}':")
        print(f"      Shape: {spatial_data.shape}")
        print(f"      Data type: {spatial_data.dtype}")
        print(f"      Sample values: {spatial_data[:3]}")
        
        if spatial_data.shape[1] != 2:
            print(f"      ⚠️  WARNING: Expected 2D coordinates, got {spatial_data.shape[1]}D")
        else:
            print(f"      ✅ Correct 2D spatial coordinates")
    
    # Check required fields for ResolVI
    print(f"\n🧬 RESOLVI REQUIRED FIELDS CHECK:")
    
    required_fields = {
        'Subtype': 'labels_key',
        'batch': 'batch_key', 
        'mapped_batch': 'perturbation_key',
        'p14_status': 'analysis filtering'
    }
    
    for field, purpose in required_fields.items():
        if field in adata.obs.columns:
            unique_values = adata.obs[field].unique()
            print(f"   ✅ '{field}' ({purpose}): {len(unique_values)} categories")
            if len(unique_values) < 10:  # Show categories if not too many
                print(f"      Categories: {list(unique_values)}")
        else:
            print(f"   ❌ Missing '{field}' ({purpose})")
    
    # Check for counts layer
    print(f"\n📋 EXPRESSION DATA CHECK:")
    if 'counts' in adata.layers:
        print(f"   ✅ 'counts' layer found: {adata.layers['counts'].shape}")
    else:
        print(f"   ❌ 'counts' layer missing")
        print(f"   Available layers: {list(adata.layers.keys())}")
    
    return adata, spatial_candidates

def create_test_setup(adata, spatial_key='spatial'):
    """
    Create a test setup to see if ResolVI initialization works
    """
    print(f"\n🧪 TESTING RESOLVI SETUP...")
    print("=" * 50)
    
    try:
        # Add path to your scvi-tools
        sys.path.insert(0, 'src')
        import scvi.external.resolvi as RESOLVI
        print("✅ Successfully imported RESOLVI")
        
        # Test setup_anndata
        print(f"\n📋 Testing setup_anndata with spatial_key='{spatial_key}'...")
        
        RESOLVI.RESOLVI.setup_anndata(
            adata, 
            labels_key="Subtype",
            layer="counts",
            batch_key="batch", 
            perturbation_key='mapped_batch', 
            control_perturbation='B6, IgG',
            spatial_key=spatial_key,  # This is crucial!
            background_key=None
        )
        print("✅ setup_anndata completed successfully")
        
        # Test model initialization
        print(f"\n🏗️  Testing model initialization...")
        
        spatial_resolvi = RESOLVI.RESOLVI(
            adata,
            semisupervised=True,
            n_latent=32,
            perturbation_hidden_dim=128,
            n_input_spatial=2,  # This should work now
            control_penalty_weight=1.0
        )
        print("✅ Model initialized successfully!")
        
        # Verify spatial encoder
        has_spatial_encoder = hasattr(spatial_resolvi.module.model, 'spatial_encoder')
        print(f"✅ Model has spatial encoder: {has_spatial_encoder}")
        
        return spatial_resolvi
        
    except Exception as e:
        print(f"❌ Error during setup/initialization: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Your exact data path
    path_to_query_adata = "/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/final_object_corrected.h5ad"
    
    # Diagnose the data
    result = diagnose_spatial_data(path_to_query_adata)
    
    if result is not None:
        adata, spatial_candidates = result
        
        # If spatial data exists, test the setup
        if spatial_candidates:
            # Use the first available spatial key
            spatial_key = spatial_candidates[0]
            
            # If not 'spatial', copy it to 'spatial' for consistency
            if spatial_key != 'spatial':
                print(f"\n🔄 Copying '{spatial_key}' to 'spatial' key...")
                adata.obsm['spatial'] = adata.obsm[spatial_key]
                spatial_key = 'spatial'
            
            # Test the setup
            model = create_test_setup(adata, spatial_key)
            
            if model is not None:
                print(f"\n🎉 SUCCESS! Your spatial ResolVI model is working correctly!")
                print(f"✅ Use spatial_key='{spatial_key}' in your setup_anndata call")
        else:
            print(f"\n❌ CANNOT PROCEED: No spatial coordinates found in your data")
            print(f"You need to add spatial coordinates to adata.obsm before using spatial ResolVI") 