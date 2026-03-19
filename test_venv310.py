#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_venv310.py.
Description:
    Test script to verify that all packages are properly installed in venv310.

Dependencies:
    • Python >= 3.10.
    • All packages from requirements.txt.

Usage:
    python test_venv310.py
"""

import sys
import traceback

def test_imports():
    """Test importing all key packages."""
    print("Testing package imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import skimage
        print(f"✓ Scikit-image {skimage.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-image import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import anndata
        print(f"✓ AnnData {anndata.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ AnnData import failed: {e}")
        return False
    
    try:
        import scanpy
        print(f"✓ Scanpy {scanpy.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Scanpy import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key packages."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ NumPy array creation: {arr}")
    except Exception as e:
        print(f"✗ NumPy functionality test failed: {e}")
        return False
    
    try:
        import torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"✓ PyTorch tensor creation: {tensor}")
        print(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"✗ PyTorch functionality test failed: {e}")
        return False
    
    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✓ Pandas DataFrame creation: {df.shape}")
    except Exception as e:
        print(f"✗ Pandas functionality test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("PYTHON 3.10 VIRTUAL ENVIRONMENT VERIFICATION")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("="*60)
    
    # Test imports.
    imports_ok = test_imports()

    # Test basic functionality.
    functionality_ok = test_basic_functionality()
    
    print("\n" + "="*60)
    if imports_ok and functionality_ok:
        print("✓ ALL TESTS PASSED! Virtual environment is working correctly.")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED! Please check the error messages above.")
        sys.exit(1)
