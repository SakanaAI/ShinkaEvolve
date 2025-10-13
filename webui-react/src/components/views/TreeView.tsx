import { useEffect, useRef } from 'react';
import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';
import * as d3 from 'd3';
import type { Program } from '@/types/program';

interface TreeNode extends d3.HierarchyNode<Program> {
  x: number;
  y: number;
}

export function TreeView() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { selectedDbPath, selectedNodeId, setSelectedNodeId } = useStore();
  const { data: programs, isLoading, error } = usePrograms(selectedDbPath);

  useEffect(() => {
    if (!programs || programs.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    // Create hierarchy from programs
    const root = createHierarchy(programs);

    if (!root) return;

    // Create tree layout
    const treeLayout = d3.tree<Program>().size([height - 100, width - 200]);
    const treeData = treeLayout(root) as TreeNode;

    // Create SVG group for zooming/panning
    const g = svg
      .append('g')
      .attr('transform', 'translate(100,50)');

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Draw links
    g.selectAll('.link')
      .data(treeData.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('fill', 'none')
      .attr('stroke', '#ccc')
      .attr('stroke-width', 2)
      .attr('d', d3.linkHorizontal<any, TreeNode>()
        .x((d) => d.y)
        .y((d) => d.x)
      );

    // Draw nodes
    const nodes = g
      .selectAll('.node')
      .data(treeData.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d) => `translate(${d.y},${d.x})`)
      .style('cursor', 'pointer')
      .on('click', (_event, d) => {
        setSelectedNodeId(d.data.id);
      });

    // Add circles
    nodes
      .append('circle')
      .attr('r', 6)
      .attr('fill', (d) => {
        if (!d.data.correct) return '#e74c3c';
        return getScoreColor(d.data.combined_score);
      })
      .attr('stroke', (d) => (d.data.id === selectedNodeId ? '#ff8c00' : '#666'))
      .attr('stroke-width', (d) => (d.data.id === selectedNodeId ? 3 : 2));

    // Add labels
    nodes
      .append('text')
      .attr('dy', -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#444')
      .text((d) => d.data.metadata.patch_name || d.data.id.substring(0, 8));

  }, [programs, selectedNodeId, setSelectedNodeId]);

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        Loading tree...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center text-red-500">
        Error loading tree: {error.message}
      </div>
    );
  }

  if (!programs || programs.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        No programs to display. Select a database to view the evolution tree.
      </div>
    );
  }

  return <svg ref={svgRef} className="h-full w-full" />;
}

function createHierarchy(programs: Program[]): d3.HierarchyNode<Program> | null {
  // Find root (program with no parent)
  const root = programs.find((p) => !p.parent_id);
  if (!root) return null;

  // Build parent-to-children map
  const childrenMap = new Map<string, Program[]>();
  programs.forEach((p) => {
    if (p.parent_id) {
      if (!childrenMap.has(p.parent_id)) {
        childrenMap.set(p.parent_id, []);
      }
      childrenMap.get(p.parent_id)!.push(p);
    }
  });

  // Create hierarchy with children accessor function
  return d3.hierarchy(root, (d: Program) => childrenMap.get(d.id) || null);
}

function getScoreColor(score: number): string {
  // Simple color scale from red to green
  const normalized = Math.min(Math.max(score, 0), 1);
  const hue = normalized * 120; // 0 = red, 120 = green
  return `hsl(${hue}, 70%, 50%)`;
}
