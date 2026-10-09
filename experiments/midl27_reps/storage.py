"""Google Drive I/O for the Colab pilot: drive.mount when it works, else the Drive API.

drive.mount fails in the VS Code Colab extension and intermittently in the browser
(colabtools#5944). In API mode results live in a local folder that is restored from
Drive at start-up and synced back by sync() (call it after each stage).
"""
from __future__ import annotations

import io
import os
import shutil
import time


class Drive:
    def __init__(self, folder: str = "fastmri_artifacts", results: str = "midl27_reps_results",
                 local_out: str = "/content/midl27_reps_results"):
        self.folder, self.results = folder, results
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            self.mode = "mount"
            self.art = f"/content/drive/MyDrive/{folder}"
            self.out = f"{self.art}/{results}"
            os.makedirs(self.out, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"drive.mount failed ({type(e).__name__}) -> Drive API")
            self.mode = "api"
            from google.colab import auth
            auth.authenticate_user()
            from googleapiclient.discovery import build
            self.svc = build("drive", "v3", cache_discovery=False)
            self.art_id = self._find(folder, "root", folder=True)["id"]
            f = self._find(results, self.art_id, folder=True)
            self.out_id = f["id"] if f else self.svc.files().create(
                body={"name": results, "parents": [self.art_id],
                      "mimeType": "application/vnd.google-apps.folder"}, fields="id").execute()["id"]
            self.out = local_out
            os.makedirs(self.out, exist_ok=True)
            self._synced = {}
            self._restore()
        print(f"[storage] mode={self.mode} out={self.out}")

    # -- API helpers --------------------------------------------------------
    def _find(self, name, parent, folder=False):
        q = f"name = '{name}' and '{parent}' in parents and trashed = false"
        if folder:
            q += " and mimeType = 'application/vnd.google-apps.folder'"
        r = self.svc.files().list(q=q, fields="files(id,name,size)").execute()["files"]
        return r[0] if r else None

    def _download(self, fid, dst):
        from googleapiclient.http import MediaIoBaseDownload
        with io.FileIO(dst + ".part", "wb") as fh:
            dl = MediaIoBaseDownload(fh, self.svc.files().get_media(fileId=fid), chunksize=2 ** 26)
            done = False
            while not done:
                _, done = dl.next_chunk(num_retries=5)
        os.replace(dst + ".part", dst)

    def _restore(self):
        r = self.svc.files().list(q=f"'{self.out_id}' in parents and trashed = false",
                                  fields="files(id,name)", pageSize=1000).execute()["files"]
        for f in r:
            dst = os.path.join(self.out, f["name"])
            if not os.path.exists(dst):
                self._download(f["id"], dst)
            self._synced[dst] = (os.path.getsize(dst), os.path.getmtime(dst))
        print(f"[storage] restored {len(r)} result file(s)")

    # -- public -------------------------------------------------------------
    def get_input(self, name: str, dst: str) -> str:
        """Copy an input file from the artifacts folder (e.g. network-snapshot.pkl)."""
        if os.path.exists(dst):
            return dst
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        if self.mode == "mount":
            shutil.copy(f"{self.art}/{name}", dst)
        else:
            f = self._find(name, self.art_id)
            assert f, f"{name} not in {self.folder}"
            self._download(f["id"], dst)
        return dst

    def sync(self, verbose: bool = False):
        if self.mode == "mount":
            return
        from googleapiclient.http import MediaFileUpload
        n = 0
        for fn in sorted(os.listdir(self.out)):
            p = os.path.join(self.out, fn)
            if not os.path.isfile(p) or fn.endswith((".part", ".tmp")):
                continue
            st = (os.path.getsize(p), os.path.getmtime(p))
            if self._synced.get(p) == st:
                continue
            media = MediaFileUpload(p, resumable=st[0] > 2 ** 22)
            f = self._find(fn, self.out_id)
            if f:
                self.svc.files().update(fileId=f["id"], media_body=media).execute(num_retries=5)
            else:
                self.svc.files().create(body={"name": fn, "parents": [self.out_id]}, media_body=media,
                                        fields="id").execute(num_retries=5)
            self._synced[p] = st
            n += 1
        if verbose or n:
            print(f"[storage] synced {n} file(s) -> Drive/{self.folder}/{self.results}")
