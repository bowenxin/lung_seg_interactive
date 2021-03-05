function out = dir2(addr)
%% list directory without "." and ".."
    files = char(addr); % convert potential string to char
    files = dir(files);
    files = files(~ismember({files.name},{'.', '..','.DS_Store'}));
    if nargout
        out = string({files.name})';
    else
        disp(char(files.name));
    end
end