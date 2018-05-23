# -*- mode: python -*-
import sys
sys.setrecursionlimit(10000)

block_cipher = None

added_files = [
	( 'main.ui','.'),
	( 'DataSelect.ui','.'),
	('*.png','.')
	]

a = Analysis(['ItATMISapp.py'],
             pathex=['/home/jmj136/deep-learning/ItATMIS2'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='ItATMIS',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
