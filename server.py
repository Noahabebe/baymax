from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance
import nltk
import subprocess
import os
import re
import requests
import webview
import asyncio
from ollama import AsyncClient

app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Create the webview window
window = webview.create_window('Baymax', app)

# Define command descriptions
cmd_commands = {
    "Append": "The append command can be used by programs to open files in another directory as if they were located in the current directory. The append command is available in MS-DOS as well as in all 32-bit versions of Windows. The append command is not available in 64-bit versions of Windows.",
    "Arp": "The arp command is used to display or change entries in the ARP cache. The arp command is available in all versions of Windows.",
    "Assoc": "The assoc command is used to display or change the file type associated with a particular file extension. The assoc command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "At": "The at command is used to schedule commands and other programs to run at a specific date and time. The at command is available in Windows 7, Windows Vista, and Windows XP. Beginning in Windows 8, command line task scheduling should instead be completed with the schtasks command.",
    "Atmadm": "The atmadm command is used to display information related to asynchronous transfer mode (ATM) connections on the system. The atmadm command is available in Windows XP. Support for ATM was removed beginning in Windows Vista, making the atmadm command unnecessary.",
    "Attrib": "The attrib command is used to change the attributes of a single file or a directory. The attrib command is available in all versions of Windows, as well as in MS-DOS.",
    "Auditpol": "The auditpol command is used to display or change audit policies. The auditpol command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Bcdboot": "The bcdboot command is used to copy boot files to the system partition and to create a new system BCD store. The bcdboot command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Bcdedit": "The bcdedit command is used to view or make changes to Boot Configuration Data. The bcdedit command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista. The bcdedit command replaced the bootcfg command beginning in Windows Vista.",
    "Bdehdcfg": "The bdehdcfg command is used to prepare a hard drive for BitLocker Drive Encryption. The bdehdcfg command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Bitsadmin": "The bitsadmin command is used to create, manage, and monitor download and upload jobs. The bitsadmin command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista. While the bitsadmin command is available in those versions of Windows, it is being phased out—the BITS PowerShell cmdlets should be used instead.",
    "Bootcfg": "The bootcfg command is used to build, modify, or view the contents of the boot.ini file, a hidden file that is used to identify in what folder, on which partition, and on which hard drive Windows is located. The bootcfg command is available in Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP. The bootcfg command was replaced by the bcdedit command beginning in Windows Vista. Bootcfg is still available in Windows 10, 8, 7, and Vista, but it serves no real value since boot.ini is not used in these operating systems.",
    "Bootsect": "The bootsect command is used to configure the master boot code to one compatible with BOOTMGR (Vista and later) or NTLDR (XP and earlier). The bootsect command is available in Windows 11, Windows 10, and Windows 8. The bootsect command is also available in Windows 7 and Windows Vista but only from the Command Prompt available in System Recovery Options.",
    "Break": "The break command sets or clears extended CTRL+C checking on DOS systems. The break command is available in all versions of Windows, as well as in MS-DOS. The break command is available in Windows XP and later versions of Windows to provide compatibility with MS-DOS files but it has no effect in Windows itself.",
    "Cacls": "The cacls command is used to display or change access control lists of files. The cacls command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP. The cacls command is being phased out in favor of the icacls command, which should be used instead in all versions of Windows after Windows XP.",
    "Call": "The call command is used to run a script or batch program from within another script or batch program. The call command is available in all versions of Windows, as well as in MS-DOS. The call command has no effect outside of a script or batch file. In other words, running the call command at the Command Prompt or MS-DOS prompt will do nothing.",
    "Cd": "The cd command is the shorthand version of the chdir command. The cd command is available in all versions of Windows, as well as in MS-DOS.",
    "Certreq": "The certreq command is used to perform various certification authority (CA) certificate functions. The certreq command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Certutil": "The certutil command is used to dump and display certification authority (CA) configuration information in addition to other CA functions. The certutil command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Change": "The change command changes various terminal server settings like install modes, COM port mappings, and logons. The change command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Chcp": "The chcp command displays or configures the active code page number. The chcp command is available in all versions of Windows, as well as in MS-DOS.",
    "Chdir": "The chdir command is used to display the drive letter and folder that you are currently in. Chdir can also be used to change the drive and/or directory that you want to work in. The chdir command is available in all versions of Windows, as well as in MS-DOS.",
    "Checknetisolation": "The checknetisolation command is used to test apps that require network capabilities. The checknetisolation command is available in Windows 11, Windows 10, and Windows 8.",
    "Chglogon": "The chglogon command enables, disables, or drains terminal server session logins. The chglogon command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista. Executing the chglogon command is the same as executing change logon.",
    "Chgport": "The chgport command can be used to display or change COM port mappings for DOS compatibility. The chgport command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista. Executing the chgport command is the same as executing change port.",
    "Chgusr": "The chgusr command is used to change the install mode for the terminal server. The chgusr command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista. Executing the chgusr command is the same as executing change user.",
    "Chkdsk": "The chkdsk command, often referred to as check disk, is used to identify and correct certain hard drive errors. The chkdsk command is available in all versions of Windows, as well as in MS-DOS.",
    "Chkntfs": "The chkntfs command is used to configure or display the checking of the disk drive during the Windows boot process. The chkntfs command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Choice": "The choice command is used within a script or batch program to provide a list of choices and return the value of that choice to the program. The choice command is available in MS-DOS and all versions of Windows except Windows XP. Use the set command with the /p switch in place of the choice command in batch files and scripts that you plan to use in Windows XP.",
    "Cipher": "The cipher command shows or changes the encryption status of files and folders on NTFS partitions. The cipher command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Clip": "The clip command is used to redirect the output from any command to the clipboard in Windows. The clip command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Cls": "The cls command clears the screen of all previously entered commands and other text. The cls command is available in all versions of Windows, as well as in MS-DOS.",
    "Cmd": "The cmd command starts a new instance of the cmd.exe command interpreter. The cmd command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Cmdkey": "The cmdkey command is used to show, create, and remove stored user names and passwords. The cmdkey command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Cmstp": "The cmstp command installs or uninstalls a Connection Manager service profile. The cmstp command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Color": "The color command is used to change the colors of the text and background within the Command Prompt window. The color command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Command": "The command command starts a new instance of the command.com command interpreter. The command command is available in MS-DOS as well as in all 32-bit versions of Windows. The command command is not available in 64-bit versions of Windows.",
    "Comp": "The comp command is used to compare the contents of two files or sets of files. The comp command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Compact": "The compact command is used to show or change the compression state of files and directories on NTFS partitions. The compact command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Convert": "The convert command is used to convert FAT or FAT32 formatted volumes to the NTFS format. The convert command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Copy": "The copy command does simply that — it copies one or more files from one location to another. The copy command is available in all versions of Windows, as well as in MS-DOS. The xcopy command is considered to be a more 'powerful' version of the copy command.",
    "Cscript": "The cscript command is used to execute scripts via Microsoft Script Host. The cscript command is available in all versions of Windows. The cscript command is most popularly used to manage printers from the command line using scripts like prncnfg.vbs, prndrvr.vbs, prnmngr.vbs, and others.",
    "Ctty": "The ctty command is used to change the default input and output devices for the system. The ctty command is available in Windows 98 and 95 as well as in MS-DOS. The functions provided by the ctty command were no longer necessary beginning in Windows XP because the command.com interpreter (MS-DOS) is no longer the default command line interpreter.",
    "Date": "The date command is used to show or change the current date. The date command is available in all versions of Windows, as well as in MS-DOS.",
    "Dblspace": "The dblspace command is used to create or configure DoubleSpace compressed drives. The dblspace command is available in Windows 98 and 95, as well as in MS-DOS. DriveSpace, executed using the drvspace command, is an updated version of DoubleSpace. Windows began handling compression beginning in Windows XP.",
    "Debug": "The debug command starts Debug, a command line application used to test and edit programs. The debug command is available in MS-DOS as well as in all 32-bit versions of Windows. The debug command is not available in 64-bit versions of Windows.",
    "Defrag": "The defrag command is used to defragment a drive you specify. The defrag command is the command line version of Microsoft's Disk Defragmenter. The defrag command is available in all versions of Windows, as well as in MS-DOS.",
    "Del": "The del command is used to delete one or more files. The del command is available in all versions of Windows, as well as in MS-DOS. The del command is the same as the erase command.",
    "Deltree": "The deltree command is used to delete a directory and all the files and subdirectories within it. The deltree command is available in Windows 98 and 95, as well as in MS-DOS. Beginning in Windows XP, a folder and its files and subfolders can be removed using the /s function of the rmdir command. Deltree was no longer needed with this new rmdir ability so the command was removed.",
    "Diantz": "The diantz command is used to losslessly compress one or more files. The diantz command is sometimes called Cabinet Maker. The diantz command is available in Windows 7, Windows Vista, and Windows XP. The diantz command is the same as the makecab command.",
    "Dir": "The dir command is used to display a list of files and folders contained inside the folder that you are currently working in. The dir command also displays other important information like the hard drive's serial number, the total number of files listed, their combined size, the total amount of free space left on the drive, and more. The dir command is available in all versions of Windows, as well as in MS-DOS.",
    "Diskcomp": "The diskcomp command is used to compare the contents of two floppy disks. The diskcomp command is available in all versions of Windows, as well as in MS-DOS, with the exclusion of Windows 11 and Windows 10.",
    "Diskcopy": "The diskcopy command is used to copy the entire contents of one floppy disk to another. The diskcopy command is available in all versions of Windows, as well as in MS-DOS, with the exclusion of Windows 11 and Windows 10.",
    "Diskpart": "The diskpart command is used to create, manage, and delete hard drive partitions. The diskpart command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP. The diskpart command replaced the fdisk command beginning in Windows XP.",
    "Diskperf": "The diskperf command is used to manage disk performance counters remotely. The diskperf command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Diskraid": "The diskraid command starts the DiskRAID tool which is used to manage and configure RAID arrays. The diskraid command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Dism": "The dism command starts the Deployment Image Servicing and Management tool (DISM). The DISM tool is used to manage features in Windows images. The dism command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Dispdiag": "The dispdiag command is used to output a log of information about the display system. The dispdiag command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Djoin": "The djoin command is used to create a new computer account in a domain. The djoin command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Doskey": "The doskey command is used to edit command lines, create macros, and recall previously entered commands. The doskey command is available in all versions of Windows, as well as in MS-DOS.",
    "Dosshell": "The dosshell command starts DOS Shell, a graphical file management tool for MS-DOS. The dosshell command is available in Windows 95 (in MS-DOS mode) and also in MS-DOS version 6.0 and later MS-DOS versions that were upgraded from previous versions that contained the dosshell command. A graphical file manager, Windows Explorer, became an integrated part of the operating system beginning in Windows 95.",
    "Dosx": "The dosx command is used to start DOS Protected Mode Interface (DPMI), a special mode designed to give MS-DOS applications access to more than the normally allowed 640 KB. The dosx command is available in Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP. The dosx command is not available in 64-bit versions of Windows. The dosx command and DPMI is only available in Windows to support older MS-DOS programs.",
    "Driverquery": "The driverquery command is used to show a list of all installed drivers. The driverquery command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Drvspace": "The drvspace command is used to create or configure DriveSpace compressed drives. The drvspace command is available in Windows 98 and 95, as well as in MS-DOS. DriveSpace is an updated version of DoubleSpace, executed using the dblspace command. Windows began handling compression beginning in Windows XP.",
    "Echo": "The echo command is used to show messages, most commonly from within script or batch files. The echo command can also be used to turn the echoing feature on or off. The echo command is available in all versions of Windows, as well as in MS-DOS.",
    "Edit": "The edit command starts the MS-DOS Editor tool which is used to create and modify text files. The edit command is available in MS-DOS as well as in all 32-bit versions of Windows. The edit command is not available in 64-bit versions of Windows.",
    "Edlin": "The edlin command starts the Edlin tool which is used to create and modify text files from the command line. The edlin command is available in all 32-bit versions of Windows but is not available in 64-bit versions of Windows. In MS-DOS, the edlin command is only available up to MS-DOS 5.0, so unless your later version of MS-DOS was upgraded from 5.0 or prior, you won't see the edlin command.",
    "Elsent": "The elsent command is used to configure the default handler for file types that don't have a handler defined. The elsent command is available in Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Endlocal": "The endlocal command is used to end the localization of environment changes in a batch or script file. The endlocal command is available in all versions of Windows, as well as in MS-DOS.",
    "Erase": "The erase command is used to delete one or more files. The erase command is the same as the del command. The erase command is available in all versions of Windows, as well as in MS-DOS.",
    "Esentutl": "The esentutl command is used to manage Extensible Storage Engine databases. The esentutl command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Eventcreate": "The eventcreate command is used to create a custom event in an event log. The eventcreate command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Eventquery": "The eventquery command is used to display a filtered list of events in the event logs. The eventquery command is available in Windows 7, Windows Vista, and Windows XP.",
    "Eventtriggers": "The eventtriggers command is used to configure and display event triggers on local or remote machines. The eventtriggers command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Evntcmd": "The evntcmd command is used to start the Event Viewer, a Microsoft Management Console (MMC) snap-in tool used to view events in the logs. The evntcmd command is available in Windows 98, 95, and MS-DOS.",
    "Exe2bin": "The exe2bin command is used to convert .exe files to binary format. The exe2bin command is available in MS-DOS as well as in all 32-bit versions of Windows. The exe2bin command is not available in 64-bit versions of Windows.",
    "Exit": "The exit command is used to end the Command Prompt session that you're currently working in. The exit command is available in all versions of Windows, as well as in MS-DOS.",
    "Expand": "The expand command is used to extract the files and folders contained in Microsoft Cabinet (CAB) files. The expand command is available in all versions of Windows, as well as in MS-DOS.",
    "Extrac32": "The extrac32 command is used to extract the files and folders contained in Microsoft Cabinet (CAB) files. The extrac32 command is actually a CAB extraction program for use by Internet Explorer but can be used to extract any Microsoft Cabinet file. The extrac32 command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Fc": "The fc command is used to compare two individual or sets of files and then show the differences between them. The fc command is available in all versions of Windows, as well as in MS-DOS.",
    "Fdisk": "The fdisk command is used to create, manage, and delete hard drive partitions. The fdisk command is available in MS-DOS as well as in all 32-bit versions of Windows. The fdisk command is not available in 64-bit versions of Windows. The diskpart command replaced the fdisk command beginning in Windows XP.",
    "Find": "The find command is used to search for a specified text string in one or more files. The find command is available in all versions of Windows, as well as in MS-DOS.",
    "Findstr": "The findstr command is used to search for text string patterns in one or more files. The findstr command is available in all versions of Windows, as well as in MS-DOS.",
    "Finger": "The finger command is used to return information about one or more users on a remote computer that's running the Finger service. The finger command is not available by default in Windows but can be enabled by turning on the Optional Feature called 'Services for Unix'. The finger command is available in Unix based operating systems and is used to return information about users on a remote system.",
    "For": "The for command is used to run a specified command for each file in a set of files. The for command is most often used within a batch or script file. The for command is available in all versions of Windows, as well as in MS-DOS.",
    "Forfiles": "The forfiles command selects one or more files to execute a specified command on. The forfiles command is most often used within a batch or script file. The forfiles command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Format": "The format command is used to format a drive in the file system that you specify. The format command is available in all versions of Windows, as well as in MS-DOS.",
    "Fsutil": "The fsutil command is used to perform various FAT and NTFS file system tasks like managing reparse points and sparse files, dismounting a volume, and extending a volume. The fsutil command is available in all versions of Windows, as well as in MS-DOS.",
    "Ftp": "The ftp command can be used to transfer files to and from another computer over a LAN or the Internet. The ftp command is available in all versions of Windows, as well as in MS-DOS.",
    "Ftype": "The ftype command is used to define a default program to open a specified file type. The ftype command is available in all versions of Windows, as well as in MS-DOS.",
    "Getmac": "The getmac command is used to display the media access control (MAC) address of all the network controllers on a system. The getmac command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Goto": "The goto command is used in a batch or script file to direct the command process to a labeled line in the script. The goto command is available in all versions of Windows, as well as in MS-DOS.",
    "Gpresult": "The gpresult command is used to display Group Policy settings. The gpresult command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Gpupdate": "The gpupdate command is used to update Group Policy settings. The gpupdate command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Grant": "The grant command is used to give one or more users access to a specific command in MS-DOS. The grant command is not available in any 64-bit version of Windows.",
    "Graftabl": "The graftabl command is used to enable the ability of Windows to display an extended character set in graphics mode. The graftabl command is available in Windows 98 and 95 as well as in MS-DOS.",
    "Graphics": "The graphics command is used to load a program that can print graphics. The graphics command is not available in 64-bit versions of Windows.",
    "Help": "The help command provides more detailed information on other Command Prompt commands. The help command is available in all versions of Windows, as well as in MS-DOS.",
    "Hostname": "The hostname command displays the name of the current host. The hostname command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Icacls": "The icacls command is used to display or change access control lists of files. The icacls command is an updated version of the cacls command. The icacls command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "If": "The if command is used to perform conditional functions in a batch file. The if command is available in all versions of Windows, as well as in MS-DOS.",
    "Ifshlp": "The ifshlp.sys command is a real mode driver for MS-DOS, which is required for the proper operation of certain commands, such as the share and unshare commands. The ifshlp.sys command is not available in 64-bit versions of Windows.",
    "Import": "The import command is used to copy images from a camera or other recording device to a computer. The import command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Inuse": "The inuse command is used to replace files that are currently being used by the operating system. The inuse command is available in Windows 98 and 95 as well as in MS-DOS.",
    "Ipconfig": "The ipconfig command is used to display detailed IP information for each network adapter utilizing TCP/IP. The ipconfig command can also be used to release and renew IP addresses on systems configured to receive them via a DHCP server. The ipconfig command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Isacls": "The isacls command is used to display or change discretionary access control lists (DACLs) of files. The isacls command is only available in Windows 10, Windows 8, and Windows 7.",
    "Label": "The label command is used to manage the volume label of a disk. The label command is available in all versions of Windows, as well as in MS-DOS.",
    "Lodctr": "The lodctr command is used to update registry values related to performance counters. The lodctr command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Logman": "The logman command is used to create and manage Event Trace Session and Performance logs. The logman command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Logoff": "The logoff command is used to terminate a session. The logoff command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Lpq": "The lpq command displays the status of a print queue on a computer running Line Printer Daemon (LPD). The lpq command is available in Unix based operating systems and is used to manage printer jobs in a printer queue.",
    "Lpr": "The lpr command is used to send a file to a computer running Line Printer Daemon (LPD). The lpr command is available in Unix based operating systems and is used to manage printer jobs and to copy files.",
    "Lpq": "The lpq command displays the status of a print queue on a computer running Line Printer Daemon (LPD). The lpq command is available in Unix based operating systems and is used to manage printer jobs in a printer queue.",
    "Lpr": "The lpr command is used to send a file to a computer running Line Printer Daemon (LPD). The lpr command is available in Unix based operating systems and is used to manage printer jobs and to copy files.",
    "Lpq": "The lpq command displays the status of a print queue on a computer running Line Printer Daemon (LPD). The lpq command is available in Unix based operating systems and is used to manage printer jobs in a printer queue.",
    "Lpr": "The lpr command is used to send a file to a computer running Line Printer Daemon (LPD). The lpr command is available in Unix based operating systems and is used to manage printer jobs and to copy files.",
    "Makecab": "The makecab command is used to losslessly compress one or more files. The makecab command is sometimes called Cabinet Maker. The makecab command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP. The makecab command is the same as the diantz command.",
    "Md": "The md command is the shorthand version of the mkdir command. The md command is available in all versions of Windows, as well as in MS-DOS.",
    "Mem": "The mem command shows information about used and free memory areas and programs that are currently loaded into memory in the MS-DOS subsystem. The mem command is available in MS-DOS as well as in all 32-bit versions of Windows. The mem command is not available in 64-bit versions of Windows.",
    "Mkdir": "The mkdir command is used to create a new folder. The mkdir command is the shorthand version of the mkdir command. The mkdir command is available in all versions of Windows, as well as in MS-DOS.",
    "Mklink": "The mklink command is used to create a symbolic link. The mklink command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Mode": "The mode command is used to configure system devices, most often COM and LPT ports. The mode command is available in all versions of Windows, as well as in MS-DOS.",
    "More": "The more command is used to display the information contained in a text file. The more command can also be used to paginate the results of any other Command Prompt command. The more command is available in all versions of Windows, as well as in MS-DOS.",
    "Mount": "The mount command is used to mount Network File System (NFS) network shares. The mount command is available in Windows 10, Windows 8, and Windows 7.",
    "Move": "The move command is used to move one or files from one folder to another. The move command is also used to rename directories. The move command is available in all versions of Windows, as well as in MS-DOS.",
    "Mrinfo": "The mrinfo command is used to provide information about a router's interfaces and neighbors. The mrinfo command is available in Windows 7, Windows Vista, and Windows XP.",
    "Msg": "The msg command is used to send a message to a user. The msg command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Msiexec": "The msiexec command is used to start Windows Installer, a tool used to install and configure software. The msiexec command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Msmq": "The msmq command is used to start, stop, and manage Message Queuing service components. The msmq command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Mstsc": "The mstsc command is used to open Remote Desktop Connection, a Windows utility that allows you to connect to remote computers. The mstsc command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Mu": "The mu command is used to display, add, and remove Update List (.ul) files. The mu command is available in Windows 98 and 95.",
    "Nbtstat": "The nbtstat command is used to show TCP/IP information and other statistical information about a remote computer. The nbtstat command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Net": "The net command is used to display, configure, and correct a wide variety of network settings. The net command is available in all versions of Windows, as well as in MS-DOS.",
    "Net1": "The net1 command is used to display, configure, and correct a wide variety of network settings. The net1 command is available in all versions of Windows, as well as in MS-DOS.",
    "Netcfg": "The netcfg command is used to install the Windows Preinstallation Environment (WinPE), a lightweight version of Windows used to deploy workstations. The netcfg command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Netsh": "The netsh command is used to start Network Shell, a command-line utility used to manage the network configuration of the local, or a remote, computer. The netsh command is available in all versions of Windows, as well as in MS-DOS.",
    "Netstat": "The netstat command is used to display networking statistics. The netstat command is available in all versions of Windows, as well as in MS-DOS.",
    "Nlsfunc": "The nlsfunc command is used to load information specific to a particular country or region. The nlsfunc command is available in all versions of Windows, as well as in MS-DOS.",
    "Nltest": "The nltest command is used to test secure channels between Windows computers in a domain and between domain controllers that are trusting other domains. The nltest command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Nmake": "The nmake command is used to start Microsoft Program Maintenance Utility, a tool used to build projects created by the Program Maintenance Utility utility. The nmake command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Nslookup": "The nslookup command is used to obtain domain name or IP address mapping information or to diagnose DNS-related problems. The nslookup command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Ntbackup": "The ntbackup command is used to perform various backup functions from the Command Prompt or from within a batch or script file. The ntbackup command is not available in any 64-bit version of Windows, including Windows XP Professional x64 Edition.",
    "Ntsd": "The ntsd command is used to perform certain command line debugging tasks. The ntsd command is available in Windows 7 and Windows Vista.",
    "Nul": "The nul command is used to create a bit constant of zero length. The nul device is used to discard the output of a command. The nul command is available in all versions of Windows, as well as in MS-DOS.",
    "Openfiles": "The openfiles command is used to display and disconnect open files and folders on a system. The openfiles command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Path": "The path command is used to display or set a specific path available to executable files. The path command is available in all versions of Windows, as well as in MS-DOS.",
    "Pathping": "The pathping command functions much like the tracert command but will also report information about network latency and loss at each hop. The pathping command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Pause": "The pause command is used within a batch or script file to pause the processing of the file. When the pause command is used, a Press any key to continue... message displays in the command window. The pause command is available in all versions of Windows, as well as in MS-DOS.",
    "Pentnt": "The pentnt command is used to detect floating point division errors in the Intel Pentium chip. The pentnt command is available in Windows 95, Windows 98, and Windows ME. The pentnt command is not available in 64-bit versions of Windows.",
    "Ping": "The ping command is used to test the connectivity and latency of a network connection. The ping command is available in all versions of Windows, as well as in MS-DOS.",
    "Pkgmgr": "The pkgmgr command is used to start the Windows Package Manager from the Command Prompt. The pkgmgr command is available in Windows 7 and Windows Vista.",
    "Pmon": "The pmon command starts Performance Monitor, a tool used to display performance counter data in real time. The pmon command is available in Windows 98 and 95.",
    "Popd": "The popd command is used to change the current directory to the one most recently stored by the pushd command. The popd command is most often utilized from within a batch or script file. The popd command is available in all versions of Windows, as well as in MS-DOS.",
    "Portcls": "The portcls command is used to install Port Class (PortCls) audio drivers on a system during the text-mode phase of Windows setup. The portcls command is available in Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Powercfg": "The powercfg command is used to manage the Windows power management settings from the command line. The powercfg command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Print": "The print command is used to print a specified text file to a specified printing device. The print command is available in all versions of Windows, as well as in MS-DOS.",
    "Prompt": "The prompt command is used to customize the appearance of the prompt text in Command Prompt or MS-DOS. The prompt command is available in all versions of Windows, as well as in MS-DOS.",
    "Pushd": "The pushd command is used to store a directory for use, most commonly from within a batch or script program. The pushd command is most often utilized from within a batch or script file. The pushd command is available in all versions of Windows, as well as in MS-DOS.",
    "Qappsrv": "The qappsrv command is used to display all Remote Desktop Session Host servers available on the network. The qappsrv command is available in Windows 7 and Windows Vista.",
    "Qbasic": "The qbasic command starts QBasic, the MS-DOS based programming environment for the BASIC programming language. The qbasic command is not available in any 64-bit version of Windows.",
    "Qprocess": "The qprocess command is used to display information about running processes. The qprocess command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Query": "The query command is used to display the status of a specified service. The query command is available in Windows 7, Windows Vista, and Windows XP.",
    "Quser": "The quser command is used to display information about users currently logged on to the system. The quser command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Qwinsta": "The qwinsta command is used to display information about open Remote Desktop Sessions. The qwinsta command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Rasautou": "The rasautou command is used to manage Remote Access Dialer AutoDial addresses. The rasautou command is available in Windows 8, Windows 7, and Windows Vista.",
    "Rasdial": "The rasdial command is used to start or end a network connection for a Microsoft client. The rasdial command is available in Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Rcp": "The rcp command is used to copy files between a Windows computer and a system running the rshd daemon. The rcp command is available in Windows 98 and 95.",
    "Rd": "The rd command is the shorthand version of the rmdir command. The rd command is available in all versions of Windows, as well as in MS-DOS.",
    "Recall": "The recall command is used to recall a previous command from a batch file or in the current session. The recall command is available in all versions of Windows, as well as in MS-DOS.",
    "Reg": "The reg command is used to manage the Windows Registry from the command line. The reg command is available in all versions of Windows, as well as in MS-DOS.",
    "Regsvr32": "The regsvr32 command is used to register a DLL file as a command component in the Windows Registry. The regsvr32 command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Relog": "The relog command is used to create new performance logs from data in existing performance logs. The relog command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Rem": "The rem command is used to record comments or remarks in a batch or script file. Remarks are not entered into the execution of the batch file. The rem command is available in all versions of Windows, as well as in MS-DOS.",
    "Rename": "The rename command is used to change the name of the individual file that you specify. The rename command is available in all versions of Windows, as well as in MS-DOS.",
    "Renegade": "The renegade command is used to start Microsoft Renegade, a multiplayer game that comes pre-installed on Windows 95 and 98.",
    "Replace": "The replace command is used to replace one or more files with one or more other files. The replace command is available in all versions of Windows, as well as in MS-DOS.",
    "Reset": "The reset command, executed as reset session, is used to reset the session subsystem software and hardware to known initial values. The reset command is available in Windows 8, Windows 7, and Windows Vista.",
    "Rexec": "The rexec command is used to run commands on remote computers running the rexec daemon. The rexec command is available in Windows 98 and 95 as well as in MS-DOS.",
    "Rmdir": "The rmdir command is used to delete an existing or completely empty folder. The rmdir command, also known as the rd command, is available in all versions of Windows, as well as in MS-DOS.",
    "Route": "The route command is used to manipulate network routing tables. The route command is available in all versions of Windows, as well as in MS-DOS.",
    "Rsh": "The rsh command is used to run commands on remote computers running the rsh daemon. The rsh command is available in Windows 98 and 95 as well as in MS-DOS.",
    "Rsm": "The rsm command is used to manage media resources using Removable Storage. The rsm command is available in Windows 7, Windows Vista, and Windows XP.",
    "Rwinsta": "The rwinsta command is the shorthand version of the reset session command. The rwinsta command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Sc": "The sc command is used to configure information about services. The sc command is available in all versions of Windows, as well as in MS-DOS.",
    "Schtasks": "The schtasks command is used to schedule specified programs or commands to run at certain times. The schtasks command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Sdclt": "The sdclt command is used to open Backup and Restore (Windows 7) from the Command Prompt or a shortcut. The sdclt command is available in Windows 11, Windows 10, and Windows 8.",
    "Secedit": "The secedit command is used to configure and analyze system security by comparing the current security configuration to a template. The secedit command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Set": "The set command is used to display, set, or remove environment variables in MS-DOS or from the Command Prompt. The set command is available in all versions of Windows, as well as in MS-DOS.",
    "Setlocal": "The setlocal command is used to start the localization of environment changes inside a batch or script file. The setlocal command is available in all versions of Windows, as well as in MS-DOS.",
    "Setver": "The setver command is used to set the MS-DOS version number that MS-DOS reports to a program. The setver command is not available in 64-bit versions of Windows.",
    "Sfc": "The sfc command is used to verify and replace important Windows system files. The sfc command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Shadow": "The shadow command is used to monitor another Remote Desktop Services session. The shadow command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Share": "The share command is used to install file locking and file sharing functions in MS-DOS. The share command is not available in 64-bit versions of Windows.",
    "Shift": "The shift command is used to change the position of replaceable parameters in a batch or script file. The shift command is available in all versions of Windows, as well as in MS-DOS.",
    "Shutdown": "The shutdown command is used to shut down, restart, or log off the current system or a remote computer. The shutdown command is available in all versions of Windows, as well as in MS-DOS.",
    "Sort": "The sort command is used to read data from a specified input, sort that data, and return the results of that sort to the Command Prompt screen, a file, or another output device. The sort command is available in all versions of Windows, as well as in MS-DOS.",
    "Start": "The start command is used to open a new command line window to run a specified program or command. The start command can also be used to start an application without creating a new window. The start command is available in all versions of Windows, as well as in MS-DOS.",
    "Subst": "The subst command is used to associate a local path with a drive letter. The subst command is a lot like the net use command except a local path is used instead of a shared network path. The subst command is available in all versions of Windows, as well as in MS-DOS.",
    "Sxstrace": "The sxstrace command is used to start the WinSxs Tracing Utility, a programming diagnostic tool. The sxstrace command is available in Windows 8, Windows 7, and Windows Vista.",
    "Systeminfo": "The systeminfo command is used to display basic Windows configuration information for the local or a remote computer. The systeminfo command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Taskkill": "The taskkill command is used to terminate a running task. The taskkill command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Tasklist": "The tasklist command is used to display a list of all running tasks and processes on the local or a remote computer. The tasklist command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Tcmsetup": "The tcmsetup command is used to setup or disable the Telephony Application Programming Interface (TAPI) client. The tcmsetup command is available in Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Telnet": "The telnet command is used to communicate with remote computers that use the Telnet protocol. The telnet command is available in all versions of Windows, as well as in MS-DOS.",
    "Tftp": "The tftp command is used to transfer files to and from a remote computer that's running the Trivial File Transfer Protocol (TFTP) service or daemon. The tftp command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Time": "The time command is used to show or change the current time. The time command is available in all versions of Windows, as well as in MS-DOS.",
    "Timeout": "The timeout command is typically used in a batch or script file to provide a specified timeout value during a procedure. The timeout command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Title": "The title command is used to set the Command Prompt window title. The title command is available in all versions of Windows, as well as in MS-DOS.",
    "Tlntadmn": "The tlntadmn command is used to administer a local or remote computer running Telnet Server. The tlntadmn command is not available by default in Windows 7 but can be enabled by turning on the Telnet Server Windows feature from Programs and Features in Control Panel.",
    "Tpmvscmgr": "The tpmvscmgr command is used to create, destroy, and manage TPM virtual smart cards. The tpmvscmgr command is available in Windows 8 and Windows 7.",
    "Tracerpt": "The tracerpt command is used to process event trace logs or real-time data from instrumented event trace providers and save the output in a readable text file. The tracerpt command is available in Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Tracert": "The tracert command is used to show details about the path that a packet takes to reach a destination, including where any delays occur. The tracert command is available in all versions of Windows, as well as in MS-DOS.",
    "Tree": "The tree command is used to graphically display the folder structure of a specified drive or path. The tree command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Tscon": "The tscon command is used to attach a user session to a Remote Desktop session. The tscon command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Tskill": "The tskill command is used to end the specified process. The tskill command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Tsshutdn": "The tsshutdn command is used to remotely shut down or restart a terminal server. The tsshutdn command is available in Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Type": "The type command is used to display the information contained in a text file. The type command is available in all versions of Windows, as well as in MS-DOS.",
    "Typeperf": "The typeperf command is used to display performance data in the Command Prompt window or write the data to specified log file. The typeperf command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Tzutil": "The tzutil command is used to display or configure the current system's time zone. The tzutil command is available in Windows 8, Windows 7, and Windows Vista.",
    "Umount": "The umount command is used to remove network shares from a computer's list of shared resources. The umount command is available in Windows 98 and 95 as well as in MS-DOS.",
    "Unlodctr": "The unlodctr command is used to remove Explain text and Performance counter names for a service or device driver from the Windows Registry. The unlodctr command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Ver": "The ver command is used to display the current Windows version number. The ver command is available in all versions of Windows, as well as in MS-DOS.",
    "Verify": "The verify command is used to enable or disable the ability of Command Prompt, or MS-DOS, to verify that files are written correctly to a disk. The verify command is available in MS-DOS.",
    "Vidcap": "The vidcap command is not available in any 64-bit version of Windows, including Windows XP Professional x64 Edition. The vidcap command is used to start Microsoft Video Capture, a screen and video capture tool.",
    "Vol": "The vol command shows the volume label and serial number of a specified disk, assuming this information exists. The vol command is available in all versions of Windows, as well as in MS-DOS.",
    "Vssadmin": "The vssadmin command is used to create, delete, and list information about volume shadow copies. The vssadmin command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "W32tm": "The w32tm command is used to diagnose issues with Windows Time. The w32tm command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Waitfor": "The waitfor command is used to send or wait for a signal on a system. The waitfor command is available in Windows 7, Windows Vista, and Windows XP.",
    "Wbadmin": "The wbadmin command is used to start and stop backup jobs, display details about a previous backup, list the items within a backup, and report on the status of currently running backups. The wbadmin command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Wecutil": "The wecutil command is used to manage subscriptions to events that are forwarded from WS-Management supported computers. The wecutil command is available in Windows 8, Windows 7, and Windows Vista.",
    "Wevtutil": "The wevtutil command is used to retrieve information about event logs and publishers. The wevtutil command is available in Windows 11, Windows 10, Windows 8, Windows 7, Windows Vista, and Windows XP.",
    "Where": "The where command is used to search for files that match a specified pattern. The where command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Whoami": "The whoami command is used to retrieve user name and group information on a network. The whoami command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Winrm": "The winrm command is used to start the command line tool for Windows Remote Management. The winrm command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Winrs": "The winrs command is used to open a secure command window with a specified remote computer. The winrs command is available in Windows 7 and Windows Vista.",
    "Winsat": "The winsat command starts the Windows System Assessment Tool, a program that assesses various features, attributes, and capabilities of a computer running Windows. The winsat command is available in Windows 11, Windows 10, Windows 8, Windows 7, and Windows Vista.",
    "Wmic": "The wmic command starts the Windows Management Instrumentation Command line (WMIC), a scripting interface that simplifies the use of Windows Management Instrumentation (WMI) and systems managed through WMI. The wmic command is available in Windows 11, Windows 10, Windows 8, and Windows 7.",
    "Wscript": "The wscript command is used to execute scripts via Windows Script Host. The wscript command is available in all versions of Windows, as well as in MS-DOS.",
    "Xcopy": "The xcopy command is used to copy files and directories from one location to another. The xcopy command is available from within the Command Prompt in all Windows operating systems including Windows 10, Windows 8, Windows 7, Windows Vista, Windows XP, Windows 98, and versions of Windows released before it.",
    "Xwizard": "The xwizard command is used to start the Malicious Software Removal tool GUI, a part of the Windows Malicious Software Removal Tool.",
    "Yy": "The yy command is used to read the specified input file and copy it to the specified output file, converting lowercase letters to uppercase. The yy command is available in Windows 7, Windows Vista, and Windows XP.",
    "Ztree": "The ztree command starts the ZTreeWin file manager, a text-mode file manager. The ztree command is available in Windows 10, Windows 8, Windows 7, and Windows Vista."
}

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

def calculate_similarity(description, cmd_description):
    description_tokens = preprocess_text(description)
    cmd_tokens = preprocess_text(cmd_description)
    distance = edit_distance(description_tokens, cmd_tokens)
    max_len = max(len(description_tokens), len(cmd_tokens))
    similarity = 1 - (distance / max_len)
    return similarity

def find_command(description, commands):
    best_match = None
    best_score = -1

    for command, cmd_description in commands.items():
        similarity_score = calculate_similarity(description, cmd_description)
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = command

    return best_match if best_score >= 0.1 else None

async def chats(description):
    message = {'role': 'user', 'content': description}
    async for part in await AsyncClient(host='https://noahabebe-ollama.hf.space/').chat(model='noahabebe/baymax', messages=[message], stream=True):
        return print(part['message']['content'],end='', flush=True)
         

async def commands(description, command):
    message = {'role': 'system', 'content': f"Generate a command for the following description: '{description}' using the base command: {command}. Only provide the command and arguments to run together. No description."}
    async for part in await AsyncClient(host='https://noahabebe-ollama.hf.space/').chat(model='noahabebe/baymax', messages=[message], stream=True):
        return print(part['message']['content'],end='', flush=True)
        

def get_assistant_response(description):
   
    try:
        chat = asyncio.run(chats(description))
        return chat
    except requests.exceptions.RequestException as e:
        return f"Failed to get response from assistant: {str(e)}"

def construct_command(command, description):
    
    detailed_command = asyncio.run(commands(description, command))
    
    match = re.search(r'`([^`]*)`', detailed_command)
    if match:
        actual_command = match.group(1)
    else:
        actual_command = detailed_command.strip()

    return actual_command

def execute_command(detailed_command):
    try:
        result = subprocess.run(detailed_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode()
        error = result.stderr.decode()
        if output:
            return jsonify(command=detailed_command, result=output, assistant_response="")
        if error:
            return jsonify(command=detailed_command, result=error, assistant_response="")
    except subprocess.CalledProcessError as e:
        return jsonify(command=detailed_command, result=f"Command execution failed with error: {e}", assistant_response="")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_description = request.json['description']
        matching_command = find_command(user_description, cmd_commands)

        if matching_command:
            detailed_command = construct_command(matching_command, user_description)
            result = execute_command(detailed_command)
            return result
        else:
            assistant_response = get_assistant_response(user_description)
            response = {"Baymax": assistant_response}
            return jsonify(response)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    webview.start()
