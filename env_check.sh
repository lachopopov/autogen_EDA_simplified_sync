echo "=== 1. Environment Status ==="
conda info --envs | grep ag2_env
echo "Active Python: $(which python)"
echo "Active Pip: $(which pip)"
python --version

echo -e "\n=== 2. Path Verification ==="
python -c "
import sys
print('Python executable:', sys.executable)
print('sys.path[0]:', sys.path[0])
print('site-packages in path:', any('site-packages' in p for p in sys.path))
"

echo -e "\n=== 3. AG2/Autogen Package Status ==="
pip show ag2 autogen 2>/dev/null || echo "Package missing"

echo -e "\n=== 4. Import Tests ==="
python -c "import autogen; print('✓ autogen OK:', autogen.__version__)" 2>&1 || echo "✗ autogen FAILED"
python -c "from autogen import UserProxyAgent; print('✓ UserProxyAgent OK')" 2>&1 || echo "✗ UserProxyAgent FAILED"
python -c "import ag2; print('✓ ag2 direct import OK:', ag2.__version__)" 2>&1 || echo "✗ ag2 direct import FAILED"

echo -e "\n=== 5. Dependency Conflicts ==="
pip check 2>&1 | head -10

echo -e "\n=== 6. Virtual Env Confirmation ==="
python -c "
import sys
base_prefix = getattr(sys, 'base_prefix', sys.prefix)
print('In conda env:', sys.prefix != base_prefix)
"
