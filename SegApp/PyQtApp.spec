# -*- mode: python -*-

import sys
sys.setrecursionlimit(10000)

block_cipher = None

added_files = [
         ( 'pyqt_corrector.ui', '.' ),
         ( 'pyqt_import.ui', '.' ),
		 ( 'pyqt_main.ui', '.' ),
		 ( 'pyqt_prefs.ui', '.' ),
		 ( 'UserManual.pdf', '.'),
		 ( 'QtIconFiles/*.png','QtIconFiles')
		 ('SampleDataset.mat','.')
         ]

a = Analysis(['SegApp.py'],
             pathex=['~/deep-learning/SegApp',
                    '~/deep-learning/Utils'],
             binaries=[],
             datas=added_files,
             hiddenimports=["CustomMetrics","VisTools","OverlapMetrics","h5py","h5py.defs","h5py.utils","h5py._proxy","h5py.h5ac"],
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
          exclude_binaries=True,
          name='SegApp',
          debug=True,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='SegApp')
