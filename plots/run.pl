#!/usr/bin/env perl

use strict;
use warnings;

my @nthreads = (1, 2, 4, 8, 16, 32);
my $nruns = 16;

my @args = @ARGV;

sub average{
    my($data) = @_;
    if (not @$data) {
        die("Empty arrayn");
    }
    my $total = 0;
    foreach (@$data) {
        $total += $_;
    }
    my $average = $total / @$data;
    return $average;
}
sub stdev {
    my($data) = @_;
    if(@$data == 1){
        return 0;
    }
    my $average = &average($data);
    my $sqtotal = 0;
    foreach(@$data) {
        $sqtotal += ($average-$_) ** 2;
    }
    my $std = ($sqtotal / (@$data-1)) ** 0.5;
    return $std;
}

foreach my $threads (@nthreads) {
    my @samples;
    for (my $i = 0; $i < $nruns; $i++) {
	# print "Running sample $i with threads=$threads\n";
	my $file = `mktemp`;
	chomp $file;
	print `OMP_NUM_THREADS=$threads @args 2>&1 >$file`;
	open my $fd, '<', $file;
	for (<$fd>) {
	    if (/Linear solver for poisson equation total duration: ([\d\.]+)/) {
		# print "$1\n";
		push @samples, $1;
	    }
	}
	close $fd;
    }

    my $avg = &average(\@samples);
    my $std = &stdev(\@samples);
    print "$threads $nruns $avg $std @samples\n";
}

