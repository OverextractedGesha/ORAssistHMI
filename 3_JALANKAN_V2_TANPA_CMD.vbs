Option Explicit
Dim objFSO, objShell, strScriptDir, strVenvPythonw, strAppScript, strCommand

Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

' Dapatkan direktori folder script saat ini
strScriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
strVenvPythonw = objFSO.BuildPath(strScriptDir, ".venv\Scripts\pythonw.exe")
strAppScript = objFSO.BuildPath(strScriptDir, "src\or_sterile_assist_v2.py")

' Validasi keberadaan virtual environment
If Not objFSO.FileExists(strVenvPythonw) Then
    MsgBox "Virtual Environment (.venv) belum terpasang di folder ini!" & vbCrLf & vbCrLf & _
           "Silakan jalankan file '1_SETUP_DEPENDENCIES.bat' terlebih dahulu untuk menginstall semua dependensi yang diperlukan." & vbCrLf & vbCrLf & _
           "Setelah instalasi dependensi berhasil, Anda dapat membuka file ini kembali.", _
           vbExclamation + vbOKOnly, "OR-SterileAssist v2 - Setup Diperlukan"
    WScript.Quit 1
End If

' Set working directory ke folder aplikasi
objShell.CurrentDirectory = strScriptDir

' Jalankan pythonw.exe dengan mode silent (0 = hide window, False = return immediately)
strCommand = """" & strVenvPythonw & """ """ & strAppScript & """"
objShell.Run strCommand, 0, False

Set objShell = Nothing
Set objFSO = Nothing
